"""Worker launcher for supervised swarm runs.

Spawns Claude Code or Codex CLI processes in provisioned worktrees,
reusing the managed-session wrapper so worktree locks/logs stay coherent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

from aragora.pipeline.execution_mode import ExecutionMode
from aragora.security.capability_gate import (
    Capability,
    CapabilityApprovalRequiredError,
    authorize_capability_dispatch,
    ensure_capability_approval_id,
)
from aragora.swarm.env_utils import git_safe_env
from aragora.swarm.worker_contract import build_worker_contract
from aragora.swarm.worker_process import (
    DEFAULT_VERIFICATION_TIMEOUT_SECONDS,
    LaunchConfig,
    MAX_WORKER_LOG_TAIL_CHARS,
    SESSION_ARTIFACTS,
    UTC,
    WorkerProcess,
    _SALVAGEABLE_EXIT_CODES,
    is_ignored_changed_path,
)

logger = logging.getLogger(__name__)

# Merge-gate verification should be deterministic and must not inherit the
# operator shell's live provider credentials. Tests that need keys can still
# set them explicitly inside the verification process.
_SCRUBBED_VERIFICATION_ENV_VARS: frozenset[str] = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
        "MISTRAL_API_KEY",
        "DEEPSEEK_API_KEY",
        "KIMI_API_KEY",
        "TINKER_API_KEY",
    }
)


def _strip_github_tokens(env: dict[str, str]) -> None:
    for key in (
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GH_ENTERPRISE_TOKEN",
        "GITHUB_ENTERPRISE_TOKEN",
    ):
        env.pop(key, None)


class WorkerLauncher:
    """Launch and monitor Claude Code / Codex worker processes."""

    def __init__(self, config: LaunchConfig | None = None) -> None:
        self.config = config or LaunchConfig()
        self._workers: dict[str, WorkerProcess] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._live_log_tasks: dict[str, dict[str, asyncio.Task[bytes]]] = {}
        self._live_log_handles: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _strip_session_artifacts(paths: set[str]) -> list[str]:
        """Normalize changed paths by removing harness/runtime-owned artifacts."""
        return sorted(path for path in paths if not is_ignored_changed_path(path))

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
        # Enrich work order with pre-read file context before building prompt
        work_order = self._enrich_task_context(work_order, worktree_path)
        prompt = self._build_prompt(work_order)
        session_id = str(work_order.get("owner_session_id", "")).strip()
        lease_id = str(work_order.get("lease_id", "")).strip()
        metadata = self._metadata_dict(work_order)
        admin_approved = self._is_admin_approved(work_order, metadata)
        actor_id = self._actor_id_for_work_order(work_order, metadata)
        receipt_id = self._receipt_id_for_work_order(work_order, metadata)
        approval_id, dispatch_action_id = self._authorize_worker_launch(
            work_order=work_order,
            worktree_path=worktree_path,
            agent=agent,
            actor_id=actor_id,
            receipt_id=receipt_id,
            admin_approved=admin_approved,
            metadata=metadata,
            prompt=prompt,
        )

        profile_override = str(metadata.get("claude_profile", "")).strip() or None
        profile_script_override = str(metadata.get("claude_profile_script", "")).strip() or None
        cmd = self._build_command(
            agent,
            prompt,
            worktree_path,
            session_id=session_id,
            admin_approved=admin_approved,
            claude_profile=profile_override,
            claude_profile_script=profile_script_override,
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
        raw_worker_env = metadata.get("worker_env", {}) if isinstance(metadata, dict) else {}
        worker_env_overrides = {
            str(key).strip(): str(value)
            for key, value in dict(raw_worker_env or {}).items()
            if str(key).strip()
        }
        if agent == "claude":
            effective_profile = profile_override or self.config.claude_profile
            if effective_profile and "ARAGORA_CLAUDE_PROFILE" not in worker_env_overrides:
                worker_env_overrides["ARAGORA_CLAUDE_PROFILE"] = effective_profile

        # Codex CLI multi-agent mode creates isolated config dirs that lack
        # auth credentials.  Pin CODEX_HOME to the user's main config so
        # workers can authenticate.  Claude Code doesn't use this var.
        worker_env = dict(os.environ)
        _strip_github_tokens(worker_env)
        if agent == "codex":
            codex_home = Path.home() / ".codex"
            if (codex_home / "auth.json").exists():
                worker_env["CODEX_HOME"] = str(codex_home)
        if worker_env_overrides:
            worker_env.update(worker_env_overrides)
        # Allow task-scoped overrides, but keep GitHub auth out of worker envs.
        _strip_github_tokens(worker_env)

        contract = build_worker_contract(
            agent=agent,
            config=self.config,
            worktree_path=worktree_path,
            env=worker_env,
            work_order=work_order,
        )
        contract.validate()
        contract_dict = contract.to_dict()
        contract_checksum = contract.checksum()
        work_order["worker_contract"] = dict(contract_dict)
        work_order["worker_contract_checksum"] = contract_checksum
        logger.info(
            "Worker contract: agent=%s checksum=%s",
            agent,
            contract_checksum,
        )

        # Codex uses "-" as prompt arg and reads from stdin to avoid OS
        # ARG_MAX limits on long prompts with issue bodies + file lists.
        use_stdin_prompt = agent == "codex"
        stdin_mode = asyncio.subprocess.PIPE if use_stdin_prompt else asyncio.subprocess.DEVNULL
        prompt_bytes = prompt.encode("utf-8") if use_stdin_prompt else None

        if self.config.detach:
            log_dir = Path(worktree_path)
            stdout_file = open(log_dir / ".swarm_worker_stdout.log", "w")  # noqa: SIM115
            stderr_file = open(log_dir / ".swarm_worker_stderr.log", "w")  # noqa: SIM115
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=worktree_path,
                    stdin=stdin_mode,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                    env=worker_env,
                )
                if prompt_bytes is not None and proc.stdin is not None:
                    proc.stdin.write(prompt_bytes)
                    proc.stdin.close()
            finally:
                # The subprocess inherits its own descriptors; close the
                # parent-side handles immediately to avoid ResourceWarning.
                stdout_file.close()
                stderr_file.close()
        else:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=worktree_path,
                stdin=stdin_mode,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=worker_env,
            )
            if prompt_bytes is not None and proc.stdin is not None:
                proc.stdin.write(prompt_bytes)
                proc.stdin.close()
            self._start_live_log_capture(work_order_id, worktree_path, proc)

        worker = WorkerProcess(
            work_order_id=work_order_id,
            agent=agent,
            worktree_path=worktree_path,
            branch=branch,
            pid=proc.pid,
            session_id=session_id,
            lease_id=lease_id,
            receipt_id=receipt_id,
            approval_id=approval_id,
            initial_head=initial_head,
            expected_tests=[
                str(item) for item in work_order.get("expected_tests", []) if str(item).strip()
            ],
            command=list(cmd),
            dispatch_action_id=dispatch_action_id,
            admin_approved=admin_approved,
            worker_contract=contract_dict,
            worker_contract_checksum=contract_checksum,
            prompt_chars=len(prompt),
            enriched_context_chars=len(str(work_order.get("_enriched_context", ""))),
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
        live_capture_enabled = work_order_id in self._live_log_tasks

        try:
            if live_capture_enabled:
                await asyncio.wait_for(proc.wait(), timeout=effective_timeout)
                worker.exit_code = proc.returncode
                live_logs = await self._finish_live_log_capture(
                    work_order_id,
                    worker.worktree_path,
                )
                worker.stdout = live_logs["stdout"]
                worker.stderr = live_logs["stderr"]
            else:
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
            if live_capture_enabled:
                live_logs = await self._finish_live_log_capture(
                    work_order_id,
                    worker.worktree_path,
                )
                worker.stdout = live_logs["stdout"]
                stderr_parts = [
                    part
                    for part in [
                        live_logs["stderr"].strip(),
                        f"Timed out after {effective_timeout}s",
                    ]
                    if part
                ]
                worker.stderr = "\n".join(stderr_parts)
            else:
                worker.stdout = self._read_log_file(worker.worktree_path, "stdout")
                stderr_parts = [
                    part
                    for part in [
                        self._read_log_file(worker.worktree_path, "stderr").strip(),
                        f"Timed out after {effective_timeout}s",
                    ]
                    if part
                ]
                worker.stderr = "\n".join(stderr_parts)
            logger.warning("Worker %s timed out", work_order_id)

        session_meta = self._read_session_meta(worker.worktree_path)
        session_exit_code, session_completed_at = self._terminal_session_result(session_meta)
        missing_terminal_marker = session_exit_code is None
        if session_exit_code is not None:
            worker.exit_code = session_exit_code
        worker.completed_at = datetime.now(UTC).isoformat()
        if session_completed_at:
            worker.completed_at = session_completed_at
        try:
            worker.diff = await self._collect_diff(worker.worktree_path)

            _can_query_dirty_tree = self._can_query_dirty_tree(worker)
            # ``git diff HEAD`` only detects modifications to tracked files.
            # Workers that create NEW (untracked) files show no diff.
            # Always fall back to ``git status --porcelain`` which detects
            # both untracked and modified files.
            _has_changes = bool(worker.diff) or (
                _can_query_dirty_tree and await self._has_working_tree_changes(worker.worktree_path)
            )
            if (
                self.config.auto_commit
                and not missing_terminal_marker
                and self._should_attempt_auto_commit(worker, has_changes=_has_changes)
            ):
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

            # Ensure the branch is pushed if the worker produced commits.
            if worker.commit_shas and not missing_terminal_marker:
                await self._auto_push(worker)

            if not missing_terminal_marker and self._should_run_verification(worker):
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
            cleanup_pid = self._session_owned_pid(worker.worktree_path, worker.pid, session_meta)
            if cleanup_pid is not None:
                await self._wait_for_pid_exit(cleanup_pid)
            self._cleanup_session_artifacts(worker.worktree_path)

        logger.info(
            "Worker %s completed: exit=%s commits=%d changed_paths=%d",
            work_order_id,
            worker.exit_code,
            len(worker.commit_shas),
            len(worker.changed_paths),
        )

        self._processes.pop(work_order_id, None)
        if missing_terminal_marker and worker.exit_code == 0:
            worker.exit_code = 1
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

    def collect_finished_sync(
        self,
        *,
        work_order_ids: list[str] | None = None,
    ) -> list[WorkerProcess]:
        """Collect workers that have already exited without awaiting the loop.

        This is the sync counterpart used by ``SwarmSupervisor.refresh_run()``
        when it is itself called from async code and cannot safely invoke
        ``asyncio.run()``. Only workers whose subprocess already reports a
        terminal ``returncode`` are eligible.
        """
        completed: list[WorkerProcess] = []
        ids = work_order_ids or list(self._processes.keys())
        for work_order_id in ids:
            proc = self._processes.get(work_order_id)
            worker = self._workers.get(work_order_id)
            session_exit_code: int | None = None
            if worker is not None:
                session_meta = self._read_session_meta(worker.worktree_path)
                session_exit_code, _ = self._terminal_session_result(session_meta)
                observed_pid = self._pid_for_active_lock(
                    worker.worktree_path,
                    worker.pid,
                    session_meta,
                )
                if self._active_session_lock_blocks_collection(worker.worktree_path, observed_pid):
                    continue
            if proc is None or (proc.returncode is None and session_exit_code is None):
                continue
            completed.append(self._wait_sync(work_order_id))
        return completed

    async def snapshot_progress(self, work_order: dict[str, Any]) -> dict[str, Any]:
        """Capture lightweight progress state for a dispatched worker."""
        worktree_path = str(work_order.get("worktree_path", "")).strip()
        initial_head = str(work_order.get("initial_head", "")).strip()
        pid = self._normalized_pid(work_order.get("pid"))

        snapshot: dict[str, Any] = {
            "pid_alive": False,
            "head_sha": "",
            "changed_paths": [],
            "diff_lines": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "stdout_size": 0,
            "stderr_size": 0,
            "stdout_mtime_ns": 0,
            "stderr_mtime_ns": 0,
            "has_progress_heartbeat": False,
        }
        if not worktree_path:
            return snapshot
        session_meta = self._read_session_meta(worktree_path)
        pid = self._session_owned_pid(worktree_path, pid, session_meta)
        snapshot["pid_alive"] = self._is_pid_running(pid) if pid is not None else False
        lock_pid = self._pid_for_active_lock(worktree_path, pid, session_meta)
        if self._active_session_lock_blocks_collection(worktree_path, lock_pid):
            snapshot["pid_alive"] = True

        head_sha = await self._git_output(worktree_path, "rev-parse", "HEAD")
        diff = await self._collect_diff(worktree_path)
        stdout_path = Path(worktree_path) / ".swarm_worker_stdout.log"
        stderr_path = Path(worktree_path) / ".swarm_worker_stderr.log"
        stdout_tail = self._tail_text(self._read_log_file(worktree_path, "stdout"))
        stderr_tail = self._tail_text(self._read_log_file(worktree_path, "stderr"))
        try:
            stdout_stat = stdout_path.stat()
            stdout_size = int(stdout_stat.st_size)
            stdout_mtime_ns = int(stdout_stat.st_mtime_ns)
        except OSError:
            stdout_size = 0
            stdout_mtime_ns = 0
        try:
            stderr_stat = stderr_path.stat()
            stderr_size = int(stderr_stat.st_size)
            stderr_mtime_ns = int(stderr_stat.st_mtime_ns)
        except OSError:
            stderr_size = 0
            stderr_mtime_ns = 0
        changed_paths = await self._collect_changed_paths(
            worktree_path,
            initial_head=initial_head,
            head_sha=head_sha,
        )
        # Detect progress heartbeat: stdout activity (size > 0 and recently
        # modified) signals the worker is still making progress even when no
        # git commits have landed yet.  This prevents the no-progress timeout
        # from killing workers that are reading a large codebase.
        has_progress_heartbeat = bool(stdout_size > 0 and stdout_mtime_ns > 0) or bool(
            stderr_size > 0 and stderr_mtime_ns > 0
        )
        snapshot.update(
            {
                "head_sha": head_sha,
                "changed_paths": list(changed_paths),
                "diff_lines": diff.count("\n") if diff else 0,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "stdout_size": stdout_size,
                "stderr_size": stderr_size,
                "stdout_mtime_ns": stdout_mtime_ns,
                "stderr_mtime_ns": stderr_mtime_ns,
                "has_progress_heartbeat": has_progress_heartbeat,
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
        admin_approved: bool = False,
        claude_profile: str | None = None,
        claude_profile_script: str | None = None,
    ) -> list[str]:
        """Build the launch command for the given agent type."""
        inner = self._build_agent_command(
            agent,
            prompt,
            worktree_path=worktree_path,
            admin_approved=admin_approved,
            claude_profile=claude_profile,
            claude_profile_script=claude_profile_script,
        )
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
        effective_session_id = session_id or Path(worktree_path).resolve().name
        cmd.extend(["--session-id", effective_session_id])
        cmd.append("--")
        cmd.extend(inner)
        return cmd

    def _build_agent_command(
        self,
        agent: str,
        prompt: str,
        *,
        worktree_path: str = "",
        admin_approved: bool = False,
        claude_profile: str | None = None,
        claude_profile_script: str | None = None,
    ) -> list[str]:
        if agent == "claude":
            session_mode = os.environ.get("ARAGORA_CLAUDE_SESSION_MODE", "single").strip()
            if session_mode.lower() == "multi_turn":
                prompt_path = self._write_worker_prompt(worktree_path, prompt)
                cmd = [
                    sys.executable,
                    "-m",
                    "aragora.swarm.claude_session_runner",
                    "--prompt-file",
                    prompt_path,
                    "--claude-path",
                    self.config.claude_path,
                ]
                if self.config.claude_model:
                    cmd.extend(["--model", self.config.claude_model])
                if (
                    self.config.allow_claude_dangerously_skip_permissions
                    and self.config.execution_mode == ExecutionMode.AUTONOMOUS
                ):
                    cmd.append("--dangerously-skip-permissions")
            else:
                cmd = [self.config.claude_path, "-p", prompt]
                if (
                    self.config.allow_claude_dangerously_skip_permissions
                    and self.config.execution_mode == ExecutionMode.AUTONOMOUS
                ):
                    cmd.append("--dangerously-skip-permissions")
                if self.config.claude_model:
                    cmd.extend(["--model", self.config.claude_model])
            profile = str(claude_profile or self.config.claude_profile or "").strip() or None
            if profile:
                profile_script = (
                    claude_profile_script
                    or self.config.claude_profile_script
                    or str(Path(worktree_path).resolve() / "scripts" / "claude_profile.sh")
                )
                return [profile_script, "exec", profile, "--", *cmd]
            return cmd

        if agent == "codex":
            # Use stdin ("-") for prompts to avoid OS arg length limits.
            # Long prompts with issue bodies + file lists can exceed ARG_MAX.
            cmd = [self.config.codex_path, "exec", "-"]
            if (
                self.config.allow_codex_full_auto
                and self.config.execution_mode == ExecutionMode.AUTONOMOUS
            ):
                cmd.append("--full-auto")
            if self.config.codex_model:
                cmd.extend(["--model", self.config.codex_model])
            git_dir = self._resolve_worktree_gitdir(worktree_path) if admin_approved else ""
            if git_dir:
                cmd.extend(["--add-dir", git_dir])
            return cmd

        logger.warning("Unknown agent %r, falling back to claude", agent)
        cmd = [self.config.claude_path, "-p", prompt]
        if (
            self.config.allow_claude_dangerously_skip_permissions
            and self.config.execution_mode == ExecutionMode.AUTONOMOUS
        ):
            cmd.append("--dangerously-skip-permissions")
        if self.config.claude_profile:
            profile_script = self.config.claude_profile_script or str(
                Path(worktree_path).resolve() / "scripts" / "claude_profile.sh"
            )
            return [profile_script, "exec", self.config.claude_profile, "--", *cmd]
        return cmd

    @staticmethod
    def _write_worker_prompt(worktree_path: str, prompt: str) -> str:
        prompt_dir = Path(worktree_path).resolve() / ".aragora"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = prompt_dir / "worker_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        return str(prompt_path)

    @staticmethod
    def _metadata_dict(work_order: dict[str, Any]) -> dict[str, Any]:
        metadata = work_order.get("metadata")
        return dict(metadata) if isinstance(metadata, dict) else {}

    @staticmethod
    def _strict_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        return None

    @classmethod
    def _is_admin_approved(cls, work_order: dict[str, Any], metadata: dict[str, Any]) -> bool:
        return (
            cls._strict_bool(work_order.get("admin_approved")) is True
            or cls._strict_bool(metadata.get("admin_approved")) is True
        )

    @staticmethod
    def _actor_id_for_work_order(work_order: dict[str, Any], metadata: dict[str, Any]) -> str:
        for candidate in (
            metadata.get("requested_by"),
            metadata.get("user_id"),
            work_order.get("owner_session_id"),
            work_order.get("lease_id"),
            work_order.get("work_order_id"),
            "system",
        ):
            text = str(candidate or "").strip()
            if text:
                return text
        return "system"

    @staticmethod
    def _receipt_id_for_work_order(work_order: dict[str, Any], metadata: dict[str, Any]) -> str:
        return str(
            work_order.get("receipt_id")
            or metadata.get("receipt_id")
            or metadata.get("decision_receipt_id")
            or ""
        ).strip()

    def _authorize_worker_launch(
        self,
        *,
        work_order: dict[str, Any],
        worktree_path: str,
        agent: str,
        actor_id: str,
        receipt_id: str,
        admin_approved: bool,
        metadata: dict[str, Any],
        prompt: str,
    ) -> tuple[str, str]:
        if self.config.require_explicit_approval and not self.config.use_managed_session_script:
            raise CapabilityApprovalRequiredError(
                Capability.CODE_EXEC,
                "managed session wrapper is required for code execution lanes",
            )

        # When the caller (e.g. boss loop) has already authorized the run,
        # skip the per-launch approval flow entirely.
        if not self.config.require_explicit_approval:
            return "", ""

        target_resource = str(Path(worktree_path).resolve())
        payload = {
            "work_order_id": str(work_order.get("work_order_id", "")).strip(),
            "agent": agent,
            "prompt_hash": hash(prompt),
            "expected_tests": list(work_order.get("expected_tests") or []),
            "file_scope": list(work_order.get("file_scope") or []),
        }
        approval_id = ensure_capability_approval_id(
            capability=Capability.CODE_EXEC,
            actor_id=actor_id,
            target_resource=target_resource,
            input_payload=payload,
            approval_id=str(
                metadata.get("approval_id") or work_order.get("approval_id") or ""
            ).strip(),
            receipt_id=receipt_id,
            admin_approved=admin_approved,
            approved_by=str(metadata.get("approved_by") or actor_id or "system").strip(),
            metadata={
                "work_order_id": str(work_order.get("work_order_id", "")).strip(),
                "approval_request_id": str(metadata.get("approval_request_id", "")).strip(),
            },
        )
        action = authorize_capability_dispatch(
            capability=Capability.CODE_EXEC,
            actor_id=actor_id,
            target_resource=target_resource,
            input_payload=payload,
            approval_id=approval_id,
            receipt_id=receipt_id,
            metadata={"agent": agent},
        )
        return approval_id, action.action_id

    @staticmethod
    def _resolve_worktree_gitdir(worktree_path: str) -> str:
        """Return the common git directory for a git worktree, or '' for regular repos.

        Git worktrees have a `.git` *file* (not directory) containing
        ``gitdir: <path>`` pointing to ``.git/worktrees/<name>/``.
        That worktree-specific dir has a ``commondir`` file pointing
        back to the parent ``.git/`` directory.

        The Codex ``--full-auto`` sandbox only allows writes inside the
        worktree itself.  ``git add`` needs write access to both
        ``.git/worktrees/<name>/`` (index, HEAD) and ``.git/objects/``
        (blob storage), so we return the common ``.git/`` directory to
        cover both via a single ``--add-dir``.
        """
        if not worktree_path:
            return ""
        dot_git = Path(worktree_path) / ".git"
        if not dot_git.exists() or dot_git.is_dir():
            return ""
        try:
            text = dot_git.read_text().strip()
            if not text.startswith("gitdir:"):
                return ""
            gitdir = text.split(":", 1)[1].strip()
            resolved = (dot_git.parent / gitdir).resolve()
            if not resolved.is_dir():
                return ""
            # Resolve the common directory (parent .git/) via commondir
            # file.  This covers .git/objects/, .git/refs/, and the
            # worktree-specific .git/worktrees/<name>/ subdirectory.
            commondir_file = resolved / "commondir"
            if commondir_file.is_file():
                commondir = commondir_file.read_text().strip()
                common = (resolved / commondir).resolve()
                if common.is_dir():
                    return str(common)
            # Fallback to the worktree gitdir itself if commondir
            # is missing (shouldn't happen in practice).
            return str(resolved)
        except OSError:
            pass
        return ""

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
    @staticmethod
    def _enrich_task_context(
        work_order: dict[str, Any],
        worktree_path: str,
    ) -> dict[str, Any]:
        """Read target files and related code to build rich task context.

        Design: Give the worker FULL file contents (not truncated) for focused
        tasks. Include the test file, caller context, and directory CLAUDE.md.
        This is the single highest-impact factor for worker success rate.
        """
        file_scope = work_order.get("file_scope", [])
        if not file_scope or not worktree_path:
            return work_order

        context_snippets: list[str] = []
        wt = Path(worktree_path)
        max_lines_per_file = 500  # Full content for focused files

        # 1. Read target files (full content up to 500 lines)
        for file_path in file_scope[:3]:  # Focus on top 3 files
            full_path = wt / file_path
            if not full_path.exists():
                context_snippets.append(
                    f"--- {file_path} DOES NOT EXIST ---\n"
                    "This file needs to be created as part of this task."
                )
                continue
            try:
                content = full_path.read_text(errors="replace")
                lines = content.splitlines()
                if len(lines) > max_lines_per_file:
                    snippet = "\n".join(lines[:max_lines_per_file])
                    context_snippets.append(
                        f"--- {file_path} (first {max_lines_per_file} of {len(lines)} lines) ---\n"
                        f"{snippet}\n--- end (truncated) ---"
                    )
                else:
                    context_snippets.append(
                        f"--- {file_path} ({len(lines)} lines, complete) ---\n"
                        f"{content}\n--- end ---"
                    )
            except OSError:
                pass

        # 2. Read the test file (if validation points to a test)
        expected_tests = work_order.get("expected_tests", [])
        for test_path in expected_tests[:1]:  # Include first test file
            test_full = wt / test_path
            if test_full.exists() and test_path not in file_scope:
                try:
                    test_content = test_full.read_text(errors="replace")
                    test_lines = test_content.splitlines()
                    if len(test_lines) > max_lines_per_file:
                        snippet = "\n".join(test_lines[:max_lines_per_file])
                        context_snippets.append(
                            f"--- {test_path} (test file, first {max_lines_per_file} of "
                            f"{len(test_lines)} lines) ---\n{snippet}\n--- end (truncated) ---"
                        )
                    else:
                        context_snippets.append(
                            f"--- {test_path} (test file, {len(test_lines)} lines, complete) ---\n"
                            f"{test_content}\n--- end ---"
                        )
                except OSError:
                    pass

        # 3. Find callers/importers of the target module
        for file_path in file_scope[:2]:
            full_path = wt / file_path
            if not full_path.exists():
                continue
            try:
                content = full_path.read_text(errors="replace")
                # Find key public symbols
                symbols = re.findall(r"(?:def|class)\s+(\w+)", content)
                caller_notes: list[str] = []
                for sym in symbols[:8]:
                    try:
                        result = subprocess.run(
                            ["grep", "-rn", sym, "aragora/", "--include=*.py", "-l"],
                            capture_output=True,
                            text=True,
                            cwd=worktree_path,
                            timeout=5,
                        )
                        callers = [
                            f for f in result.stdout.strip().splitlines()[:5] if f != file_path
                        ]
                        if callers:
                            caller_notes.append(f"`{sym}` used in: {', '.join(callers)}")
                    except (subprocess.TimeoutExpired, OSError):
                        pass
                if caller_notes:
                    context_snippets.append(
                        f"Cross-references for {file_path}:\n"
                        + "\n".join(f"  - {n}" for n in caller_notes)
                    )
            except OSError:
                pass

        # 4. Include directory CLAUDE.md if present
        for file_path in file_scope[:1]:
            dir_path = (wt / file_path).parent
            claude_md = dir_path / "CLAUDE.md"
            if claude_md.exists():
                try:
                    md_content = claude_md.read_text(errors="replace")
                    if len(md_content) < 3000:  # Only include if reasonably short
                        context_snippets.append(
                            f"--- {dir_path.relative_to(wt)}/CLAUDE.md (local conventions) ---\n"
                            f"{md_content}\n--- end ---"
                        )
                except OSError:
                    pass

        # 5. Recent git history for the target file
        for file_path in file_scope[:1]:
            full_path = wt / file_path
            if not full_path.exists():
                continue
            try:
                result = subprocess.run(
                    ["git", "log", "--oneline", "-5", "--", file_path],
                    capture_output=True,
                    text=True,
                    cwd=worktree_path,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    context_snippets.append(
                        f"Recent changes to {file_path}:\n{result.stdout.strip()}"
                    )
            except (subprocess.TimeoutExpired, OSError):
                pass

        if context_snippets:
            work_order = dict(work_order)
            work_order["_enriched_context"] = "\n\n".join(context_snippets)

        return work_order

    @staticmethod
    def _format_repair_journal(entries: Any, *, max_entries: int = 2, max_tail: int = 400) -> str:
        if not isinstance(entries, list) or not entries:
            return ""
        lines: list[str] = []
        for entry in entries[-max_entries:]:
            if not isinstance(entry, dict):
                continue
            at = str(entry.get("at") or "").strip()
            exit_code = entry.get("exit_code")
            worker_outcome = str(entry.get("worker_outcome") or "").strip()
            failure_reason = str(entry.get("failure_reason") or "").strip()
            header_parts = [
                part for part in [f"exit={exit_code}", worker_outcome, failure_reason] if part
            ]
            header = f"- Attempt {at}: " if at else "- Attempt: "
            header += ", ".join(header_parts) if header_parts else "details"
            lines.append(header)

            failing = entry.get("failing_verification")
            if isinstance(failing, dict):
                cmd = str(failing.get("command", "")).strip()
                if cmd:
                    lines.append(
                        f"  - failing verification: {cmd} (exit {failing.get('exit_code')})"
                    )
                stderr_tail = str(failing.get("stderr_tail", "")).strip()
                if stderr_tail:
                    lines.append(f"  - stderr: {stderr_tail[-max_tail:]}")
                stdout_tail = str(failing.get("stdout_tail", "")).strip()
                if stdout_tail:
                    lines.append(f"  - stdout: {stdout_tail[-max_tail:]}")
            else:
                stderr_tail = str(entry.get("stderr_tail", "")).strip()
                if stderr_tail:
                    lines.append(f"  - stderr: {stderr_tail[-max_tail:]}")
                stdout_tail = str(entry.get("stdout_tail", "")).strip()
                if stdout_tail:
                    lines.append(f"  - stdout: {stdout_tail[-max_tail:]}")

            changed_paths = entry.get("changed_paths")
            if isinstance(changed_paths, list) and changed_paths:
                lines.append(f"  - changed: {', '.join(str(p) for p in changed_paths)}")
        return "\n".join(lines).strip()

    @staticmethod
    def _build_prompt(work_order: dict[str, Any]) -> str:
        """Build the task prompt from a work order dict.

        Design philosophy: 90% context, 10% instructions. The agent is an
        intelligent collaborator — give it understanding, not constraints.
        Trust it to commit, test, and make good decisions.
        """
        parts: list[str] = []
        metadata = work_order.get("metadata", {})
        target_agent = str(work_order.get("target_agent", "")).strip().lower()

        # --- Section 1: Task goal (plain English) ---
        title = str(work_order.get("title", "")).strip()
        if title:
            parts.append(f"# {title}")

        description = str(work_order.get("description", "")).strip()
        if description:
            parts.append(description)

        # --- Section 2: Prior attempts (learning material) ---
        repair_notes = WorkerLauncher._format_repair_journal(metadata.get("repair_journal"))
        if repair_notes:
            parts.append("## What was tried before (and why it failed)\n\n" + repair_notes)

        # --- Section 3: Code context (the bulk of the prompt) ---
        enriched_context = work_order.get("_enriched_context", "")
        if enriched_context:
            parts.append(f"## Code context\n\n{enriched_context}")

        # --- Section 4: File scope (guidance, not hard boundary) ---
        file_scope = work_order.get("file_scope", [])
        if file_scope:
            scope_list = "\n".join(f"  - {f}" for f in file_scope)
            parts.append(
                f"FILE SCOPE GUIDANCE:\n"
                f"The planner expects you to work in these paths:\n{scope_list}\n"
                "IMPORTANT: Before starting, verify these paths exist. If they do not, "
                "search the codebase for the actual files that match the intent "
                "(e.g. `find . -name '*.py' | grep <keyword>`). Work on the real files "
                "you find — do not create files at non-existent paths just to satisfy "
                "the scope list. Treat the resolved scope as a hard boundary: do not "
                "modify files outside it, and if the fix genuinely requires other files, "
                "stop and report that blocker instead of widening scope."
            )

        # --- Section 5: Validation (concise) ---
        acceptance = metadata.get("acceptance_criteria", [])
        if acceptance:
            criteria_text = "\n".join(f"  - {c}" for c in acceptance)
            parts.append(f"Acceptance criteria:\n{criteria_text}")

        expected_tests = work_order.get("expected_tests", [])
        if expected_tests:
            tests_text = "\n".join(f"  - `{t}`" for t in expected_tests)
            parts.append(f"Expected validation:\n{tests_text}")

        constraints = metadata.get("constraints", [])
        if constraints:
            constraints_text = "\n".join(f"  - {c}" for c in constraints)
            parts.append(f"Constraints:\n{constraints_text}")

        # --- Section 6: Approval boundary (if needed) ---
        approval_required = bool(work_order.get("approval_required", False))
        if approval_required:
            parts.append(
                "Decision boundary:\n"
                "  - If you hit a real ambiguity, approval boundary, or blocker, "
                "stop cleanly and report the exact reason instead of widening scope."
            )

        # --- Section 7: Commit discipline (minimal, agent-aware) ---
        parts.append(
            "CRITICAL — You MUST commit your changes:\n"
            "After making changes, run:\n"
            "```\ngit add <specific-files> && git commit -m 'fix: description of changes'\n```\n"
            "If you do not commit, your work will be lost."
        )

        # Codex-specific: commit early (before validation) due to token budget
        if target_agent == "codex":
            parts.append(
                "Codex note: commit IMMEDIATELY after writing code, BEFORE running tests. "
                "If validation fails, the commit still preserves your work."
            )

        # --- Section 8: Lease tracking (one line) ---
        lease_id = str(work_order.get("lease_id", "")).strip()
        if lease_id:
            parts.append(
                "Receipt expectation:\n"
                f"  - Lease id: {lease_id}\n"
                "  - Aragora will record the completion receipt after a successful exit from this lane."
            )

        # --- Section 9: Stop condition (compact) ---
        parts.append(
            "Stop condition:\n"
            "  - Finish the bounded lane or stop at a real blocker.\n"
            "  - Run the expected validation commands when possible, or state exactly why they could not run.\n"
            "  - Stage only the files you intentionally changed with `git add <file> ...`.\n"
            "  - Do NOT use `git add -A` or `git add .` — session metadata files must not be committed.\n"
            '  - Commit with a descriptive message using `git commit -m "..."` before exiting.\n'
            "  - Push only when the lane has explicit remote-mutation approval; otherwise leave the local commit intact.\n"
            "  - If `git push` fails (e.g. no remote, permission error), that is acceptable — "
            "the harness will attempt to push for you.\n"
            "  - Exit with a truthful final state; do not claim integration or approval work is done unless it happened in this lane.\n\n"
            "REMINDER: A run that exits without a git commit is a FAILED run, even if exit code is 0. "
            "The harness only detects your deliverable via git commits. Always commit before exiting."
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

    @staticmethod
    def _git_output_sync(worktree_path: str, *args: str) -> str:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return ""
        if proc.returncode != 0:
            return ""
        return proc.stdout.rstrip()

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
            if path and not is_ignored_changed_path(path):
                return True
        return False

    @classmethod
    def _has_working_tree_changes_sync(cls, worktree_path: str) -> bool:
        status = cls._git_output_sync(
            worktree_path,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        for line in status.splitlines():
            if len(line) < 4:
                continue
            path = line[3:].strip()
            if path and not is_ignored_changed_path(path):
                return True
        return False

    @classmethod
    async def _collect_diff(cls, worktree_path: str) -> str:
        return await cls._git_output(worktree_path, "diff", "HEAD")

    @classmethod
    def _collect_diff_sync(cls, worktree_path: str) -> str:
        return cls._git_output_sync(worktree_path, "diff", "HEAD")

    @classmethod
    async def _collect_commit_shas(
        cls,
        worktree_path: str,
        *,
        initial_head: str,
        head_sha: str,
    ) -> list[str]:
        if not head_sha:
            return []

        # Primary path: compare initial_head to current HEAD
        if initial_head and initial_head != head_sha:
            output = await cls._git_output(
                worktree_path, "rev-list", "--reverse", f"{initial_head}..{head_sha}"
            )
            shas = [line.strip() for line in output.splitlines() if line.strip()]
            if shas:
                return shas

        # Fail closed when initial_head is missing or unchanged. Falling back
        # to origin/main can misattribute pre-existing branch commits to the
        # current worker when the lane starts from stale or non-main history.
        return []

    @classmethod
    def _collect_commit_shas_sync(
        cls,
        worktree_path: str,
        *,
        initial_head: str,
        head_sha: str,
    ) -> list[str]:
        if not head_sha:
            return []

        if initial_head and initial_head != head_sha:
            output = cls._git_output_sync(
                worktree_path, "rev-list", "--reverse", f"{initial_head}..{head_sha}"
            )
            shas = [line.strip() for line in output.splitlines() if line.strip()]
            if shas:
                return shas

        # Fail closed when initial_head is missing or unchanged. Falling back
        # to origin/main can misattribute pre-existing branch commits to the
        # current worker when the lane starts from stale or non-main history.
        return []

    @classmethod
    async def _collect_changed_paths(
        cls,
        worktree_path: str,
        *,
        initial_head: str,
        head_sha: str,
    ) -> list[str]:
        changed: set[str] = set()
        diff_range = ""
        if initial_head and head_sha and initial_head != head_sha:
            diff_range = f"{initial_head}..{head_sha}"
        if diff_range:
            diff_names = await cls._git_output(
                worktree_path,
                "diff",
                "--name-only",
                diff_range,
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
        return cls._strip_session_artifacts(changed)

    @classmethod
    def _collect_changed_paths_sync(
        cls,
        worktree_path: str,
        *,
        initial_head: str,
        head_sha: str,
    ) -> list[str]:
        changed: set[str] = set()
        diff_range = ""
        if initial_head and head_sha and initial_head != head_sha:
            diff_range = f"{initial_head}..{head_sha}"
        if diff_range:
            diff_names = cls._git_output_sync(
                worktree_path,
                "diff",
                "--name-only",
                diff_range,
            )
            changed.update(line.strip() for line in diff_names.splitlines() if line.strip())

        status_output = cls._git_output_sync(
            worktree_path,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        for line in status_output.splitlines():
            if not line or len(line) < 2:
                continue
            if len(line) > 2 and line[2] == " ":
                path = line[3:].strip()
            else:
                first_space = line.find(" ")
                if first_space < 0:
                    continue
                path = line[first_space + 1 :].strip()
            if " -> " in path:
                path = path.split(" -> ")[-1].strip()
            if path:
                changed.add(path)
        return cls._strip_session_artifacts(changed)

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
        allow_session_meta_pid_fallback: bool = True,
    ) -> WorkerProcess | None:
        """Collect results from a detached worker by checking PID and worktree state.

        Returns None if the worker is still running (PID alive).
        Returns a WorkerProcess with collected results if finished.
        """
        session_meta = cls._read_session_meta(worktree_path)
        session_exit_code, session_completed_at = cls._terminal_session_result(session_meta)
        observed_pid = cls._normalized_pid(pid)
        if allow_session_meta_pid_fallback:
            observed_pid = cls._session_owned_pid(worktree_path, observed_pid, session_meta)
        lock_pid = (
            cls._pid_for_active_lock(worktree_path, observed_pid, session_meta)
            if allow_session_meta_pid_fallback
            else observed_pid
        )
        missing_terminal_marker = session_exit_code is None
        cleanup_artifacts = True
        preserve_terminal_evidence = False
        should_honor_active_lock = allow_session_meta_pid_fallback or observed_pid is not None
        if should_honor_active_lock and cls._active_session_lock_blocks_collection(
            worktree_path, lock_pid
        ):
            return None
        elif (
            missing_terminal_marker
            and observed_pid is not None
            and cls._is_pid_running(observed_pid)
        ):
            return None

        worker = WorkerProcess(
            work_order_id=work_order_id,
            agent=agent,
            worktree_path=worktree_path,
            branch=branch,
            pid=observed_pid,
            initial_head=initial_head,
            expected_tests=[
                str(item).strip() for item in expected_tests or [] if str(item).strip()
            ],
            worker_contract={},
            worker_contract_checksum="",
        )

        worker.stdout = cls._read_log_file(worktree_path, "stdout")
        worker.stderr = cls._read_log_file(worktree_path, "stderr")
        worker.exit_code = 0 if session_exit_code is None else session_exit_code
        worker.completed_at = session_completed_at or datetime.now(UTC).isoformat()
        try:
            worker.diff = await cls._collect_diff(worktree_path)

            _can_query_dirty_tree = cls._can_query_dirty_tree(worker)
            # ``git diff HEAD`` only detects modifications to tracked files.
            # Workers that create NEW (untracked) files show no diff.
            # Always fall back to ``git status --porcelain`` which detects
            # both untracked and modified files.
            _has_changes = bool(worker.diff) or (
                _can_query_dirty_tree and await cls._has_working_tree_changes(worktree_path)
            )
            # Without a terminal session marker, treat dirty-tree state as
            # evidence only. Auto-committing here can manufacture a synthetic
            # deliverable from a partial run.
            if (
                auto_commit
                and not missing_terminal_marker
                and cls._should_attempt_auto_commit(worker, has_changes=_has_changes)
            ):
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

            # Ensure the branch is pushed if the worker produced commits.
            # _auto_push only runs inside _auto_commit which is skipped when the
            # worker already committed on its own.
            if worker.commit_shas and not missing_terminal_marker:
                await cls._auto_push(worker)

            if not missing_terminal_marker and cls._should_run_verification(worker):
                worker.verification_results = await cls._run_verification_commands(
                    worktree_path,
                    worker.expected_tests,
                )
                worker.tests_run = [
                    str(item.get("command", "")).strip()
                    for item in worker.verification_results
                    if str(item.get("command", "")).strip()
                ]
            if missing_terminal_marker and not worker.commit_shas and not worker.changed_paths:
                preserve_terminal_evidence = True
        finally:
            if cleanup_artifacts and not preserve_terminal_evidence:
                # Wait for the worker process (including its shell EXIT trap) to
                # fully terminate before removing artifacts.  Without this wait
                # the codex_session.sh trap can recreate .codex_session_meta.json
                # and append to .codex_session.log after Python-side cleanup (#902).
                _cleanup_pid = observed_pid
                if _cleanup_pid is not None:
                    await cls._wait_for_pid_exit(_cleanup_pid)
                cls._cleanup_session_artifacts(worktree_path)

        logger.info(
            "Collected detached worker %s: commits=%d changed_paths=%d",
            work_order_id,
            len(worker.commit_shas),
            len(worker.changed_paths),
        )
        if missing_terminal_marker:
            # A dead detached PID with no terminal session marker should not be
            # reported as a clean success. Only surface it if there is concrete
            # work to salvage; otherwise let the supervisor classify it as
            # "without receipt or exit marker". Preserve the session artifacts
            # in that case so the supervisor can still snapshot the worker's
            # logs before deciding how to escalate the lane.
            if not worker.commit_shas and not worker.changed_paths:
                return None
            worker.exit_code = 1
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
    def _normalized_pid(raw_pid: Any) -> int | None:
        if isinstance(raw_pid, bool):
            return None
        if isinstance(raw_pid, int):
            pid = raw_pid
        elif isinstance(raw_pid, str):
            text = raw_pid.strip()
            if not text or not re.fullmatch(r"[0-9]+", text):
                return None
            try:
                pid = int(text)
            except ValueError:
                return None
        else:
            return None
        return pid if pid > 0 else None

    @classmethod
    def _authoritative_session_pid(
        cls, pid: int | None, session_meta: dict[str, Any]
    ) -> int | None:
        """Prefer the harness-owned session PID over stale caller metadata."""
        meta_pid = cls._normalized_pid(session_meta.get("pid"))
        if meta_pid is not None:
            return meta_pid
        return pid

    @classmethod
    def _session_lock_pid_groups(cls, worktree_path: str) -> tuple[list[int], list[int]]:
        active_lock = Path(worktree_path) / ".codex_session_active"
        try:
            raw = active_lock.read_text(encoding="utf-8")
        except OSError:
            return [], []
        session_pids: list[int] = []
        parent_pids: list[int] = []
        for line in raw.splitlines():
            entry = line.strip()
            if not entry:
                continue
            if "=" not in entry:
                pid = cls._normalized_pid(entry)
                # Older lockfiles sometimes used a bare "1" sentinel to mean
                # "lock exists" without recording a usable PID. Only trust
                # bare numeric lines when they look like real session PIDs.
                if pid is not None and pid > 1 and pid not in session_pids:
                    session_pids.append(pid)
                continue
            key, value = entry.split("=", 1)
            normalized_key = key.strip()
            if normalized_key not in {"pid", "ppid"}:
                continue
            pid = cls._normalized_pid(value.strip())
            if pid is None:
                continue
            target = session_pids if normalized_key == "pid" else parent_pids
            if pid not in target:
                target.append(pid)
        return session_pids, parent_pids

    @classmethod
    def _session_lock_pids(cls, worktree_path: str) -> list[int]:
        session_pids, parent_pids = cls._session_lock_pid_groups(worktree_path)
        # Prefer the harness session PID over any optional parent PID entries.
        # The parent may outlive the managed session briefly and must not
        # become the authoritative liveness/cleanup target just because it was
        # listed first in the lock file.
        return session_pids + [pid for pid in parent_pids if pid not in session_pids]

    @classmethod
    def _session_owned_pid(
        cls,
        worktree_path: str,
        pid: int | None,
        session_meta: dict[str, Any],
    ) -> int | None:
        meta_pid = cls._normalized_pid(session_meta.get("pid"))
        if meta_pid is not None:
            return meta_pid
        lock_pids = cls._session_lock_pids(worktree_path)
        if lock_pids:
            return lock_pids[0]
        return pid

    @classmethod
    def _active_session_lock_blocks_collection(cls, worktree_path: str, pid: int | None) -> bool:
        active_lock = Path(worktree_path) / ".codex_session_active"
        if not active_lock.exists():
            return False
        session_pids, parent_pids = cls._session_lock_pid_groups(worktree_path)
        if session_pids:
            return any(cls._is_pid_running(lock_pid) for lock_pid in session_pids)
        if parent_pids:
            return any(cls._is_pid_running(lock_pid) for lock_pid in parent_pids)
        # codex_session.sh writes ended_at/exit_code before it removes the
        # active lock in its EXIT trap. Treat the lock as authoritative while
        # it still exists unless the session PID is clearly gone.
        if pid is None:
            return True
        return cls._is_pid_running(pid)

    @classmethod
    def _pid_for_active_lock(
        cls,
        worktree_path: str,
        pid: int | None,
        session_meta: dict[str, Any],
    ) -> int | None:
        """Return the harness-owned PID that should qualify an active-session lock."""
        active_lock = Path(worktree_path) / ".codex_session_active"
        if not active_lock.exists():
            return pid
        return cls._session_owned_pid(worktree_path, pid, session_meta)

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
        raw_ended_at = session_meta.get("ended_at")
        if not isinstance(raw_ended_at, str):
            return None, None
        ended_at = raw_ended_at.strip()
        if not ended_at:
            return None, None
        raw_exit_code = session_meta.get("exit_code")
        if isinstance(raw_exit_code, bool):
            return None, ended_at
        if isinstance(raw_exit_code, int):
            exit_code = raw_exit_code
        elif isinstance(raw_exit_code, str):
            text = raw_exit_code.strip()
            if not text or not re.fullmatch(r"-?[0-9]+", text):
                return None, ended_at
            try:
                exit_code = int(text)
            except ValueError:
                return None, ended_at
        else:
            return None, ended_at
        return exit_code, ended_at

    @classmethod
    async def _collect_staged_session_artifact_paths(cls, worktree_path: str) -> list[str]:
        staged_output = await cls._git_output(worktree_path, "diff", "--cached", "--name-only")
        return [
            path
            for raw_path in staged_output.splitlines()
            if (path := raw_path.strip()) and Path(path).name in SESSION_ARTIFACTS
        ]

    @classmethod
    def _collect_staged_session_artifact_paths_sync(cls, worktree_path: str) -> list[str]:
        staged_output = cls._git_output_sync(worktree_path, "diff", "--cached", "--name-only")
        return [
            path
            for raw_path in staged_output.splitlines()
            if (path := raw_path.strip()) and Path(path).name in SESSION_ARTIFACTS
        ]

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
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
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
        verification_env = cls._verification_environment(worktree_path)
        for raw_command in commands:
            command = str(raw_command).strip()
            if not command:
                continue
            execution_command = cls._prepare_verification_command(command)

            started = time.monotonic()
            try:
                proc = await asyncio.create_subprocess_exec(
                    "/bin/bash",
                    "-lc",
                    execution_command,
                    cwd=worktree_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=verification_env,
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

    @classmethod
    def _run_verification_commands_sync(
        cls,
        worktree_path: str,
        commands: list[str],
        *,
        timeout: float = DEFAULT_VERIFICATION_TIMEOUT_SECONDS,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        verification_env = cls._verification_environment(worktree_path)
        for raw_command in commands:
            command = str(raw_command).strip()
            if not command:
                continue
            execution_command = cls._prepare_verification_command(command)

            started = time.monotonic()
            try:
                proc = subprocess.run(
                    ["/bin/bash", "-lc", execution_command],
                    cwd=worktree_path,
                    capture_output=True,
                    text=True,
                    env=verification_env,
                    timeout=timeout,
                    check=False,
                )
                exit_code = proc.returncode
                stdout = proc.stdout
                stderr = proc.stderr
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
            except subprocess.TimeoutExpired:
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
    def _runtime_repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _ensure_live_node_modules(cls, worktree_root: Path) -> Path | None:
        runtime_node_modules = cls._runtime_repo_root() / "aragora" / "live" / "node_modules"
        if runtime_node_modules.is_dir():
            return runtime_node_modules
        return None

    @staticmethod
    def _prepend_env_path(env: dict[str, str], key: str, entries: list[Path]) -> None:
        values = [str(entry) for entry in entries if str(entry).strip()]
        existing = env.get(key, "").strip()
        if existing:
            values.extend(part for part in existing.split(os.pathsep) if part)
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        if deduped:
            env[key] = os.pathsep.join(deduped)

    @classmethod
    def _verification_environment(cls, worktree_path: str) -> dict[str, str]:
        worktree_root = Path(worktree_path).resolve()
        env = dict(os.environ)
        for key in _SCRUBBED_VERIFICATION_ENV_VARS:
            env.pop(key, None)

        python_entries = [worktree_root]
        debate_src = worktree_root / "aragora-debate" / "src"
        if debate_src.is_dir():
            python_entries.append(debate_src)
        cls._prepend_env_path(env, "PYTHONPATH", python_entries)

        node_modules = cls._ensure_live_node_modules(worktree_root)
        if node_modules is not None:
            cls._prepend_env_path(env, "NODE_PATH", [node_modules])
            bin_dir = node_modules / ".bin"
            if bin_dir.is_dir():
                cls._prepend_env_path(env, "PATH", [bin_dir])
        return env

    @staticmethod
    def _normalize_verification_command(command: str) -> str:
        """Rewrite leading ``python`` to the current interpreter.

        Verification commands are executed under ``bash -lc`` and therefore
        inherit the caller's PATH. On this machine ``python`` can resolve to
        a non-executable AWS CLI shim, which breaks merge-gate verification
        even when the worker produced a valid deliverable. Preserve the
        original command for reporting, but execute with a stable interpreter.
        """
        match = re.match(r"(?P<prefix>\s*)python(?=\s|$)", command)
        if not match:
            return command
        prefix = match.group("prefix")
        return f"{prefix}{shlex.quote(sys.executable)}{command[match.end() :]}"

    @staticmethod
    def _pytest_command_args(command: str) -> list[str]:
        text = str(command or "").strip()
        if not text:
            return []
        try:
            tokens = shlex.split(text)
        except ValueError:
            return []
        if len(tokens) >= 3 and tokens[1] == "-m" and tokens[2] == "pytest":
            return tokens[3:]
        if tokens and tokens[0].endswith("pytest"):
            return tokens[1:]
        return []

    @classmethod
    def _prepare_verification_command(cls, command: str) -> str:
        normalized = cls._normalize_verification_command(command).strip()
        pytest_args = cls._pytest_command_args(normalized)
        if not pytest_args:
            return normalized
        serialized_args = ", ".join(repr(arg) for arg in pytest_args)
        return (
            f"{shlex.quote(sys.executable)} - <<'PY'\n"
            "import pytest\n"
            f"raise SystemExit(pytest.main([{serialized_args}]))\n"
            "PY"
        )

    @staticmethod
    def _can_query_dirty_tree(worker: WorkerProcess) -> bool:
        return worker.exit_code == 0 or worker.exit_code in _SALVAGEABLE_EXIT_CODES

    @staticmethod
    def _should_attempt_auto_commit(worker: WorkerProcess, *, has_changes: bool) -> bool:
        if not has_changes:
            return False
        return worker.exit_code == 0 or worker.exit_code in _SALVAGEABLE_EXIT_CODES

    @staticmethod
    def _should_run_verification(worker: WorkerProcess) -> bool:
        if not worker.expected_tests:
            return False
        return worker.exit_code == 0 or worker.exit_code in _SALVAGEABLE_EXIT_CODES

    @staticmethod
    def _read_log_file(worktree_path: str, stream: str) -> str:
        """Read a detached worker's log file (stdout or stderr)."""
        log_path = Path(worktree_path) / f".swarm_worker_{stream}.log"
        try:
            return log_path.read_text(errors="replace")
        except (FileNotFoundError, OSError):
            return ""

    def _start_live_log_capture(
        self,
        work_order_id: str,
        worktree_path: str,
        proc: asyncio.subprocess.Process,
    ) -> None:
        tasks: dict[str, asyncio.Task[bytes]] = {}
        handles: dict[str, Any] = {}
        for stream_name in ("stdout", "stderr"):
            stream = getattr(proc, stream_name, None)
            if not isinstance(stream, asyncio.StreamReader):
                continue
            log_path = Path(worktree_path) / f".swarm_worker_{stream_name}.log"
            handle = open(log_path, "wb")  # noqa: SIM115
            handles[stream_name] = handle
            tasks[stream_name] = asyncio.create_task(
                self._tee_stream_to_log(stream, handle, stream_name=stream_name)
            )
        if tasks:
            self._live_log_tasks[work_order_id] = tasks
            self._live_log_handles[work_order_id] = handles
            return

        for handle in handles.values():
            handle.close()

    async def _finish_live_log_capture(
        self,
        work_order_id: str,
        worktree_path: str,
    ) -> dict[str, str]:
        tasks = self._live_log_tasks.pop(work_order_id, {})
        handles = self._live_log_handles.pop(work_order_id, {})
        captured: dict[str, str] = {}
        try:
            for stream_name in ("stdout", "stderr"):
                task = tasks.get(stream_name)
                if task is None:
                    captured[stream_name] = self._read_log_file(worktree_path, stream_name)
                    continue
                try:
                    data = await task
                except Exception:
                    logger.debug(
                        "Attached %s capture failed for %s",
                        stream_name,
                        work_order_id,
                        exc_info=True,
                    )
                    captured[stream_name] = self._read_log_file(worktree_path, stream_name)
                    continue
                captured[stream_name] = data.decode(errors="replace")
        finally:
            for handle in handles.values():
                try:
                    handle.close()
                except OSError:
                    logger.debug("Failed to close live log handle", exc_info=True)
        return captured

    def _finish_live_log_capture_sync(
        self,
        work_order_id: str,
        worktree_path: str,
    ) -> dict[str, str]:
        tasks = self._live_log_tasks.pop(work_order_id, {})
        handles = self._live_log_handles.pop(work_order_id, {})
        captured: dict[str, str] = {}
        try:
            for stream_name in ("stdout", "stderr"):
                task = tasks.get(stream_name)
                if task is not None and task.done() and not task.cancelled():
                    try:
                        captured[stream_name] = task.result().decode(errors="replace")
                        continue
                    except Exception:
                        logger.debug(
                            "Sync attached %s capture failed for %s",
                            stream_name,
                            work_order_id,
                            exc_info=True,
                        )
                captured[stream_name] = self._read_log_file(worktree_path, stream_name)
        finally:
            for handle in handles.values():
                try:
                    handle.close()
                except OSError:
                    logger.debug("Failed to close sync live log handle", exc_info=True)
        return captured

    @staticmethod
    async def _tee_stream_to_log(
        stream: asyncio.StreamReader,
        handle: Any,
        *,
        stream_name: str,
    ) -> bytes:
        captured = bytearray()
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                break
            captured.extend(chunk)
            try:
                handle.write(chunk)
                handle.flush()
            except OSError:
                logger.debug("Failed to write %s live log chunk", stream_name, exc_info=True)
        return bytes(captured)

    @staticmethod
    def _tail_text(text: str, *, max_chars: int = MAX_WORKER_LOG_TAIL_CHARS) -> str:
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    @staticmethod
    def _worker_actor_id(worker: WorkerProcess) -> str:
        for candidate in (worker.session_id, worker.lease_id, worker.work_order_id, "system"):
            text = str(candidate or "").strip()
            if text:
                return text
        return "system"

    @classmethod
    def _authorize_worker_capability(
        cls,
        worker: WorkerProcess,
        *,
        capability: Capability,
        input_payload: Any,
        approval_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        actor_id = cls._worker_actor_id(worker)
        target_resource = str(Path(worker.worktree_path).resolve())
        effective_approval_id = ensure_capability_approval_id(
            capability=capability,
            actor_id=actor_id,
            target_resource=target_resource,
            input_payload=input_payload,
            approval_id=approval_id,
            receipt_id=worker.receipt_id,
            admin_approved=worker.admin_approved,
            approved_by=actor_id,
            metadata=metadata,
        )
        action = authorize_capability_dispatch(
            capability=capability,
            actor_id=actor_id,
            target_resource=target_resource,
            input_payload=input_payload,
            approval_id=effective_approval_id,
            receipt_id=worker.receipt_id,
            metadata=metadata,
        )
        return effective_approval_id, action.action_id

    @staticmethod
    async def _auto_commit(worker: WorkerProcess) -> None:
        """Auto-commit changes in the worktree, excluding session artifacts."""
        try:
            approval_id, _ = WorkerLauncher._authorize_worker_capability(
                worker,
                capability=Capability.GIT_WRITE,
                input_payload={
                    "agent": worker.agent,
                    "work_order_id": worker.work_order_id,
                    "branch": worker.branch,
                },
                approval_id=worker.git_write_approval_id,
                metadata={"work_order_id": worker.work_order_id},
            )
            worker.git_write_approval_id = approval_id
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

            # Unstage staged session artifacts by basename so nested harness
            # metadata cannot slip through as a deliverable.
            for artifact_path in await WorkerLauncher._collect_staged_session_artifact_paths(
                worker.worktree_path
            ):
                reset_proc = await asyncio.create_subprocess_exec(
                    "git",
                    "reset",
                    "HEAD",
                    "--",
                    artifact_path,
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
        except CapabilityApprovalRequiredError as exc:
            logger.info(
                "Auto-commit skipped for %s: %s — working tree left intact",
                worker.work_order_id,
                exc.reason,
            )
        except (asyncio.TimeoutError, FileNotFoundError, OSError) as exc:
            logger.warning("Auto-commit failed for %s: %s", worker.work_order_id, exc)

    @staticmethod
    def _auto_commit_sync(worker: WorkerProcess) -> None:
        """Sync auto-commit used by pre-reap refresh paths."""
        try:
            approval_id, _ = WorkerLauncher._authorize_worker_capability(
                worker,
                capability=Capability.GIT_WRITE,
                input_payload={
                    "agent": worker.agent,
                    "work_order_id": worker.work_order_id,
                    "branch": worker.branch,
                },
                approval_id=worker.git_write_approval_id,
                metadata={"work_order_id": worker.work_order_id},
            )
            worker.git_write_approval_id = approval_id
            add_proc = subprocess.run(
                ["git", "add", "-A"],
                cwd=worker.worktree_path,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if add_proc.returncode != 0:
                logger.warning(
                    "sync git add -A failed for %s (rc=%s): %s",
                    worker.work_order_id,
                    add_proc.returncode,
                    add_proc.stderr.strip(),
                )
                return

            for artifact_path in WorkerLauncher._collect_staged_session_artifact_paths_sync(
                worker.worktree_path
            ):
                subprocess.run(
                    ["git", "reset", "HEAD", "--", artifact_path],
                    cwd=worker.worktree_path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )

            diff_index_proc = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=worker.worktree_path,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if diff_index_proc.returncode == 0:
                logger.info(
                    "No staged changes for %s after sync git add — skipping commit",
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
            commit_proc = subprocess.run(
                ["git", "commit", "--no-verify", "-m", msg],
                cwd=worker.worktree_path,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if commit_proc.returncode != 0:
                logger.warning(
                    "sync git commit failed for %s (rc=%s): %s",
                    worker.work_order_id,
                    commit_proc.returncode,
                    commit_proc.stderr.strip(),
                )
                return

            WorkerLauncher._auto_push_sync(worker)
        except CapabilityApprovalRequiredError as exc:
            logger.info(
                "Sync auto-commit skipped for %s: %s — working tree left intact",
                worker.work_order_id,
                exc.reason,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("Sync auto-commit failed for %s: %s", worker.work_order_id, exc)

    @staticmethod
    async def _auto_push(worker: WorkerProcess) -> None:
        """Best-effort push of the worker branch to origin.

        Called after auto-commit to ensure the deliverable is available for
        review and PR creation.  If push fails the local commit is still
        collected — the deliverable gate accepts branch+commit_shas from
        the local worktree.
        """
        try:
            approval_id, action_id = WorkerLauncher._authorize_worker_capability(
                worker,
                capability=Capability.GIT_PUSH,
                input_payload={
                    "agent": worker.agent,
                    "work_order_id": worker.work_order_id,
                    "branch": worker.branch,
                    "commit_shas": list(worker.commit_shas),
                },
                approval_id=worker.push_approval_id,
                metadata={"work_order_id": worker.work_order_id},
            )
            worker.push_approval_id = approval_id
            worker.push_action_id = action_id
            push_proc = await asyncio.create_subprocess_exec(
                "git",
                "push",
                "origin",
                "HEAD",
                cwd=worker.worktree_path,
                env=git_safe_env(),
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
        except CapabilityApprovalRequiredError as exc:
            logger.info(
                "Auto-push skipped for %s: %s — local commit preserved",
                worker.work_order_id,
                exc.reason,
            )
        except (asyncio.TimeoutError, FileNotFoundError, OSError) as exc:
            logger.info(
                "Auto-push skipped for %s: %s — local commit preserved",
                worker.work_order_id,
                exc,
            )

    @staticmethod
    def _auto_push_sync(worker: WorkerProcess) -> None:
        try:
            approval_id, action_id = WorkerLauncher._authorize_worker_capability(
                worker,
                capability=Capability.GIT_PUSH,
                input_payload={
                    "agent": worker.agent,
                    "work_order_id": worker.work_order_id,
                    "branch": worker.branch,
                    "commit_shas": list(worker.commit_shas),
                },
                approval_id=worker.push_approval_id,
                metadata={"work_order_id": worker.work_order_id},
            )
            worker.push_approval_id = approval_id
            worker.push_action_id = action_id
            push_proc = subprocess.run(
                ["git", "push", "origin", "HEAD"],
                cwd=worker.worktree_path,
                env=git_safe_env(),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if push_proc.returncode != 0:
                logger.info(
                    "Sync auto-push failed for %s (rc=%s): %s — local commit preserved",
                    worker.work_order_id,
                    push_proc.returncode,
                    push_proc.stderr.strip()[:200],
                )
            else:
                logger.info("Sync auto-pushed branch for %s", worker.work_order_id)
        except CapabilityApprovalRequiredError as exc:
            logger.info(
                "Sync auto-push skipped for %s: %s — local commit preserved",
                worker.work_order_id,
                exc.reason,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.info(
                "Sync auto-push skipped for %s: %s — local commit preserved",
                worker.work_order_id,
                exc,
            )

    @classmethod
    def _wait_for_pid_exit_sync(cls, pid: int, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not cls._is_pid_running(pid):
                return
            time.sleep(0.1)
        logger.debug("PID %d still alive after %.1fs sync wait", pid, timeout)

    def _wait_sync(self, work_order_id: str) -> WorkerProcess:
        """Synchronously finalize a worker whose subprocess already exited."""
        worker = self._workers.get(work_order_id)
        proc = self._processes.get(work_order_id)
        if worker is None or proc is None:
            raise KeyError(f"No finished worker for {work_order_id}")

        session_meta = self._read_session_meta(worker.worktree_path)
        session_exit_code, session_completed_at = self._terminal_session_result(session_meta)
        missing_terminal_marker = session_exit_code is None
        # The managed-session marker is authoritative when present: it records
        # the inner worker outcome after the shell wrapper's EXIT trap runs.
        exit_code = session_exit_code if session_exit_code is not None else proc.returncode
        if exit_code is None:
            raise KeyError(f"No finished worker for {work_order_id}")

        live_capture_enabled = work_order_id in self._live_log_tasks
        worker.exit_code = exit_code
        if live_capture_enabled:
            live_logs = self._finish_live_log_capture_sync(
                work_order_id,
                worker.worktree_path,
            )
            worker.stdout = live_logs["stdout"]
            worker.stderr = live_logs["stderr"]
        else:
            worker.stdout = self._read_log_file(worker.worktree_path, "stdout")
            worker.stderr = self._read_log_file(worker.worktree_path, "stderr")

        worker.completed_at = session_completed_at or datetime.now(UTC).isoformat()
        try:
            worker.diff = self._collect_diff_sync(worker.worktree_path)

            can_query_dirty_tree = self._can_query_dirty_tree(worker)
            has_changes = bool(worker.diff) or (
                can_query_dirty_tree and self._has_working_tree_changes_sync(worker.worktree_path)
            )
            # A bare wrapper returncode is not enough to bless dirty-tree
            # state. If the harness never wrote a terminal marker, do not
            # auto-commit partial changes into a salvageable deliverable.
            if (
                self.config.auto_commit
                and not missing_terminal_marker
                and self._should_attempt_auto_commit(worker, has_changes=has_changes)
            ):
                self._auto_commit_sync(worker)

            worker.head_sha = self._git_output_sync(worker.worktree_path, "rev-parse", "HEAD")
            worker.commit_shas = self._collect_commit_shas_sync(
                worker.worktree_path,
                initial_head=worker.initial_head,
                head_sha=worker.head_sha,
            )
            worker.changed_paths = self._collect_changed_paths_sync(
                worker.worktree_path,
                initial_head=worker.initial_head,
                head_sha=worker.head_sha,
            )

            if worker.commit_shas and not missing_terminal_marker:
                self._auto_push_sync(worker)

            if not missing_terminal_marker and self._should_run_verification(worker):
                worker.verification_results = self._run_verification_commands_sync(
                    worker.worktree_path,
                    worker.expected_tests,
                )
                worker.tests_run = [
                    str(item.get("command", "")).strip()
                    for item in worker.verification_results
                    if str(item.get("command", "")).strip()
                ]
        finally:
            cleanup_pid = self._session_owned_pid(worker.worktree_path, worker.pid, session_meta)
            if cleanup_pid is not None:
                self._wait_for_pid_exit_sync(cleanup_pid)
            self._cleanup_session_artifacts(worker.worktree_path)

        logger.info(
            "Sync-collected worker %s completed: exit=%s commits=%d changed_paths=%d",
            work_order_id,
            worker.exit_code,
            len(worker.commit_shas),
            len(worker.changed_paths),
        )

        self._processes.pop(work_order_id, None)
        if missing_terminal_marker:
            # Trust the harness session marker, not a bare subprocess return
            # code, before classifying the run as a clean success.
            worker.exit_code = 1
        return worker
