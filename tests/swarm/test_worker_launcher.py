"""Tests for WorkerLauncher — spawns and monitors CLI worker processes."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.swarm.worker_launcher import (
    LaunchConfig,
    WorkerLauncher,
    WorkerProcess,
)


class TestWorkerProcess:
    def test_defaults(self):
        wp = WorkerProcess(
            work_order_id="wo-1",
            agent="claude",
            worktree_path="/tmp/wt",
            branch="main",
        )
        assert wp.is_running is False  # no pid
        assert wp.exit_code is None

    def test_is_running_with_pid(self):
        wp = WorkerProcess(
            work_order_id="wo-1",
            agent="codex",
            worktree_path="/tmp/wt",
            branch="main",
            pid=12345,
        )
        assert wp.is_running is True

    def test_not_running_after_exit(self):
        wp = WorkerProcess(
            work_order_id="wo-1",
            agent="claude",
            worktree_path="/tmp/wt",
            branch="main",
            pid=12345,
            exit_code=0,
        )
        assert wp.is_running is False

    def test_to_dict(self):
        wp = WorkerProcess(
            work_order_id="wo-1",
            agent="claude",
            worktree_path="/tmp/wt",
            branch="feat-x",
            pid=99,
        )
        d = wp.to_dict()
        assert d["work_order_id"] == "wo-1"
        assert d["agent"] == "claude"
        assert d["pid"] == 99
        assert d["branch"] == "feat-x"


class TestLaunchConfig:
    def test_defaults(self):
        cfg = LaunchConfig()
        assert cfg.claude_path == "claude"
        assert cfg.codex_path == "codex"
        assert cfg.timeout_seconds == 2400.0
        assert cfg.no_progress_timeout_seconds == 1800.0
        assert cfg.auto_commit is True
        assert cfg.use_managed_session_script is True


class TestBuildPrompt:
    def test_basic_prompt(self):
        wo = {
            "title": "Fix auth module",
            "description": "The auth module has a race condition",
            "file_scope": ["aragora/auth/oidc.py"],
            "expected_tests": ["python -m pytest tests/auth/ -q"],
        }
        prompt = WorkerLauncher._build_prompt(wo)
        assert "# Task: Fix auth module" in prompt
        assert "race condition" in prompt
        assert "aragora/auth/oidc.py" in prompt
        assert "python -m pytest tests/auth/ -q" in prompt
        assert "git commit" in prompt

    def test_empty_work_order(self):
        prompt = WorkerLauncher._build_prompt({})
        assert "git commit" in prompt

    def test_metadata_acceptance_criteria(self):
        wo = {
            "title": "Add feature",
            "metadata": {
                "acceptance_criteria": ["All tests pass", "No regressions"],
                "constraints": ["Do not modify CLAUDE.md"],
            },
        }
        prompt = WorkerLauncher._build_prompt(wo)
        assert "All tests pass" in prompt
        assert "Do not modify CLAUDE.md" in prompt

    def test_codex_prompt_includes_lane_closure_guidance(self):
        prompt = WorkerLauncher._build_prompt(
            {
                "target_agent": "codex",
                "title": "Harden execution",
            }
        )

        assert "Codex lane discipline (CRITICAL" in prompt
        assert "IMMEDIATELY after writing" in prompt
        assert "commit first, then validate" in prompt
        assert "Do not exit 0 with staged or unstaged changes remaining." in prompt

    def test_claude_prompt_omits_codex_lane_closure_guidance(self):
        prompt = WorkerLauncher._build_prompt(
            {
                "target_agent": "claude",
                "title": "Harden execution",
            }
        )

        assert "Codex lane discipline:" not in prompt

    def test_file_scope_guidance_is_hard_boundary(self):
        prompt = WorkerLauncher._build_prompt(
            {
                "title": "Stay in scope",
                "file_scope": ["aragora/swarm/supervisor.py"],
            }
        )

        assert "Treat the resolved scope as a hard boundary" in prompt
        assert "do not modify files outside it" in prompt
        assert "stop and report that blocker" in prompt


class TestBuildCommand:
    def test_claude_command(self, monkeypatch):
        monkeypatch.setattr("aragora.swarm.worker_launcher.os.geteuid", lambda: 501)
        launcher = WorkerLauncher(LaunchConfig(claude_model="claude-opus-4-6"))
        cmd = launcher._build_command("claude", "fix bug", "/tmp/wt")
        assert cmd[0] == "bash"
        assert "scripts/codex_session.sh" in cmd[1]
        # session-id is derived from worktree basename when not provided
        idx = cmd.index("--session-id")
        assert cmd[idx + 1] == "wt"
        assert "--" in cmd
        assert "-p" in cmd
        assert "fix bug" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--model" in cmd
        assert "claude-opus-4-6" in cmd

    def test_claude_command_omits_dangerous_flag_as_root(self, monkeypatch):
        monkeypatch.setattr("aragora.swarm.worker_launcher.os.geteuid", lambda: 0)
        launcher = WorkerLauncher(LaunchConfig(claude_model="claude-opus-4-6"))
        cmd = launcher._build_command("claude", "fix bug", "/tmp/wt")
        assert "--dangerously-skip-permissions" not in cmd

    def test_claude_command_fails_closed_when_geteuid_is_unavailable(self, monkeypatch):
        monkeypatch.delattr("aragora.swarm.worker_launcher.os.geteuid", raising=False)
        launcher = WorkerLauncher(LaunchConfig(claude_model="claude-opus-4-6"))
        cmd = launcher._build_command("claude", "fix bug", "/tmp/wt")
        assert "--dangerously-skip-permissions" not in cmd

    def test_codex_command(self):
        launcher = WorkerLauncher(LaunchConfig(codex_model="o3"))
        cmd = launcher._build_command("codex", "fix bug", "/tmp/wt")
        assert cmd[0] == "bash"
        assert "exec" in cmd
        # Prompt is piped via stdin using "-" to avoid ARG_MAX limits
        assert "-" in cmd
        assert "--full-auto" in cmd
        assert "--model" in cmd
        assert "o3" in cmd

    def test_unknown_agent_falls_back_to_claude(self, monkeypatch):
        monkeypatch.setattr("aragora.swarm.worker_launcher.os.geteuid", lambda: 501)
        launcher = WorkerLauncher()
        cmd = launcher._build_command("gpt5", "do thing", "/tmp/wt")
        assert cmd[0] == "bash"
        assert "--dangerously-skip-permissions" in cmd

    def test_no_model_flag_when_none(self):
        launcher = WorkerLauncher()
        cmd = launcher._build_command("claude", "task", "/tmp/wt")
        assert "--model" not in cmd

    def test_session_id_derived_from_worktree_basename(self):
        launcher = WorkerLauncher()
        cmd = launcher._build_command("codex", "task", "/managed/swarm-abc-subtask_1")
        idx = cmd.index("--session-id")
        assert cmd[idx + 1] == "swarm-abc-subtask_1"

    def test_explicit_session_id_takes_precedence(self):
        launcher = WorkerLauncher()
        cmd = launcher._build_command(
            "codex", "task", "/managed/swarm-abc-subtask_1", session_id="custom-id"
        )
        idx = cmd.index("--session-id")
        assert cmd[idx + 1] == "custom-id"

    def test_direct_cli_command_when_session_wrapper_disabled(self):
        launcher = WorkerLauncher(LaunchConfig(use_managed_session_script=False))
        cmd = launcher._build_command("claude", "task", "/tmp/wt")
        assert cmd[0] == "claude"
        assert "-p" in cmd

    def test_direct_claude_profile_command_when_session_wrapper_disabled(self):
        launcher = WorkerLauncher(
            LaunchConfig(
                use_managed_session_script=False,
                claude_profile="max-01",
                claude_profile_script="/repo/scripts/claude_profile.sh",
            )
        )
        cmd = launcher._build_command("claude", "task", "/tmp/wt")
        assert cmd[:4] == ["/repo/scripts/claude_profile.sh", "exec", "max-01", "--"]
        assert "claude" in cmd
        assert "-p" in cmd

    def test_codex_adds_worktree_gitdir_to_sandbox(self, tmp_path: Path):
        """Codex in a worktree gets --add-dir pointing to the common .git dir."""
        wt = tmp_path / "wt"
        wt.mkdir()
        parent_git = tmp_path / "repo" / ".git"
        real_gitdir = parent_git / "worktrees" / "wt"
        real_gitdir.mkdir(parents=True)
        (real_gitdir / "commondir").write_text("../..\n")
        (wt / ".git").write_text(f"gitdir: {real_gitdir}\n")
        launcher = WorkerLauncher(LaunchConfig(use_managed_session_script=False))
        cmd = launcher._build_command("codex", "task", str(wt))
        assert "--add-dir" in cmd
        idx = cmd.index("--add-dir")
        assert cmd[idx + 1] == str(parent_git.resolve())
        assert "--full-auto" in cmd

    def test_codex_no_add_dir_for_regular_repo(self, tmp_path: Path):
        """Regular repos (.git is a directory) should NOT get --add-dir."""
        wt = tmp_path / "repo"
        wt.mkdir()
        (wt / ".git").mkdir()
        launcher = WorkerLauncher(LaunchConfig(use_managed_session_script=False))
        cmd = launcher._build_command("codex", "task", str(wt))
        assert "--add-dir" not in cmd
        assert "--full-auto" in cmd

    def test_resolve_worktree_gitdir_returns_common_dir(self, tmp_path: Path):
        """Resolves to the common .git directory, not the worktree subdir."""
        wt = tmp_path / "wt"
        wt.mkdir()
        parent_git = tmp_path / "repo" / ".git"
        real_gitdir = parent_git / "worktrees" / "wt"
        real_gitdir.mkdir(parents=True)
        (real_gitdir / "commondir").write_text("../..\n")
        (wt / ".git").write_text(f"gitdir: {real_gitdir}\n")
        assert WorkerLauncher._resolve_worktree_gitdir(str(wt)) == str(parent_git.resolve())

    def test_resolve_worktree_gitdir_fallback_without_commondir(self, tmp_path: Path):
        """Falls back to worktree gitdir when commondir file is missing."""
        wt = tmp_path / "wt"
        wt.mkdir()
        real_gitdir = tmp_path / "repo" / ".git" / "worktrees" / "wt"
        real_gitdir.mkdir(parents=True)
        (wt / ".git").write_text(f"gitdir: {real_gitdir}\n")
        assert WorkerLauncher._resolve_worktree_gitdir(str(wt)) == str(real_gitdir)

    def test_resolve_worktree_gitdir_empty(self):
        """Empty path returns empty string."""
        assert WorkerLauncher._resolve_worktree_gitdir("") == ""

    def test_resolve_worktree_gitdir_regular_repo(self, tmp_path: Path):
        """.git directory (not file) returns empty string."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        assert WorkerLauncher._resolve_worktree_gitdir(str(repo)) == ""


class TestLaunch:
    @pytest.mark.asyncio
    async def test_launch_creates_worker(self, tmp_path: Path):
        launcher = WorkerLauncher()
        mock_proc = AsyncMock()
        mock_proc.pid = 42
        worktree = tmp_path / "wt"
        (worktree / "scripts").mkdir(parents=True)
        (worktree / "scripts" / "codex_session.sh").write_text(
            "#!/usr/bin/env bash\n", encoding="utf-8"
        )

        wo = {
            "work_order_id": "wo-abc",
            "target_agent": "claude",
            "title": "Test",
            "expected_tests": ["python -m pytest tests/auth/ -q"],
        }

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch.object(WorkerLauncher, "_git_output", return_value=""),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        ):
            worker = await launcher.launch(wo, worktree_path=str(worktree), branch="feat")

        assert worker.work_order_id == "wo-abc"
        assert worker.agent == "claude"
        assert worker.pid == 42
        assert worker.expected_tests == ["python -m pytest tests/auth/ -q"]
        assert worker.tests_run == []
        assert worker.is_running

    @pytest.mark.asyncio
    async def test_launch_non_detached_sets_stdin_devnull(self, tmp_path: Path):
        """Non-detached workers must close stdin to prevent PTY/interactive stalls."""
        launcher = WorkerLauncher(LaunchConfig(detach=False))
        mock_proc = AsyncMock()
        mock_proc.pid = 99
        worktree = tmp_path / "wt"
        (worktree / "scripts").mkdir(parents=True)
        (worktree / "scripts" / "codex_session.sh").write_text(
            "#!/usr/bin/env bash\n", encoding="utf-8"
        )

        wo = {
            "work_order_id": "wo-stdin",
            "target_agent": "claude",
            "title": "Stdin guard test",
        }

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch.object(WorkerLauncher, "_git_output", return_value=""),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            await launcher.launch(wo, worktree_path=str(worktree), branch="feat")

        call_kwargs = mock_exec.call_args
        assert call_kwargs.kwargs.get("stdin") == asyncio.subprocess.DEVNULL

    @pytest.mark.asyncio
    async def test_launch_detached_codex_uses_stdin_pipe(self, tmp_path: Path):
        """Codex workers pipe prompt via stdin to avoid ARG_MAX limits."""
        launcher = WorkerLauncher(LaunchConfig(detach=True))
        mock_proc = AsyncMock()
        mock_proc.pid = 100
        mock_proc.stdin = AsyncMock()
        worktree = tmp_path / "wt"
        (worktree / "scripts").mkdir(parents=True)
        (worktree / "scripts" / "codex_session.sh").write_text(
            "#!/usr/bin/env bash\n", encoding="utf-8"
        )

        wo = {
            "work_order_id": "wo-stdin-detach",
            "target_agent": "codex",
            "title": "Stdin guard detached",
        }

        with (
            patch("shutil.which", return_value="/usr/bin/codex"),
            patch.object(WorkerLauncher, "_git_output", return_value=""),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            await launcher.launch(wo, worktree_path=str(worktree), branch="feat")

        call_kwargs = mock_exec.call_args
        assert call_kwargs.kwargs.get("stdin") == asyncio.subprocess.PIPE

    @pytest.mark.asyncio
    async def test_launch_detached_closes_parent_log_handles(self, tmp_path: Path):
        launcher = WorkerLauncher(LaunchConfig(detach=True))
        mock_proc = AsyncMock()
        mock_proc.pid = 101
        worktree = tmp_path / "wt"
        (worktree / "scripts").mkdir(parents=True)
        (worktree / "scripts" / "codex_session.sh").write_text(
            "#!/usr/bin/env bash\n", encoding="utf-8"
        )

        wo = {
            "work_order_id": "wo-detach-handles",
            "target_agent": "codex",
            "title": "Detached handle cleanup",
        }

        with (
            patch("shutil.which", return_value="/usr/bin/codex"),
            patch.object(WorkerLauncher, "_git_output", return_value=""),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            await launcher.launch(wo, worktree_path=str(worktree), branch="feat")

        stdout_handle = mock_exec.call_args.kwargs.get("stdout")
        stderr_handle = mock_exec.call_args.kwargs.get("stderr")
        assert stdout_handle is not None
        assert stderr_handle is not None
        assert stdout_handle.closed is True
        assert stderr_handle.closed is True

    @pytest.mark.asyncio
    async def test_launch_raises_on_missing_cli(self):
        launcher = WorkerLauncher()
        wo = {"work_order_id": "wo-1", "target_agent": "claude"}

        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="CLI not found"):
                await launcher.launch(wo, worktree_path="/tmp/wt")

    @pytest.mark.asyncio
    async def test_launch_merges_metadata_worker_env(self, tmp_path: Path):
        launcher = WorkerLauncher(LaunchConfig(detach=False))
        mock_proc = AsyncMock()
        mock_proc.pid = 103
        worktree = tmp_path / "wt"
        (worktree / "scripts").mkdir(parents=True)
        (worktree / "scripts" / "codex_session.sh").write_text(
            "#!/usr/bin/env bash\n", encoding="utf-8"
        )

        wo = {
            "work_order_id": "wo-env",
            "target_agent": "claude",
            "title": "Env propagation test",
            "metadata": {
                "worker_env": {
                    "ARAGORA_RELEVANT_FILES": "aragora/swarm/boss_loop.py",
                    "ARAGORA_TEST_PATTERNS": "tests/swarm/test_boss_loop.py",
                }
            },
        }

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch.object(WorkerLauncher, "_git_output", return_value=""),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            await launcher.launch(wo, worktree_path=str(worktree), branch="feat")

        env = mock_exec.call_args.kwargs.get("env")
        assert isinstance(env, dict)
        assert env["ARAGORA_RELEVANT_FILES"] == "aragora/swarm/boss_loop.py"
        assert env["ARAGORA_TEST_PATTERNS"] == "tests/swarm/test_boss_loop.py"


class TestWait:
    @pytest.mark.asyncio
    async def test_wait_collects_results(self):
        launcher = WorkerLauncher(LaunchConfig(auto_commit=False))
        expected_test = "python -m pytest tests/swarm/test_supervisor.py -q"

        # Set up a mock worker + process
        worker = WorkerProcess(
            work_order_id="wo-1",
            agent="claude",
            worktree_path="/tmp/wt",
            branch="main",
            pid=100,
            expected_tests=[expected_test],
        )
        launcher._workers["wo-1"] = worker

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output text", b""))
        mock_proc.returncode = 0
        launcher._processes["wo-1"] = mock_proc
        verification_results = [
            {
                "command": expected_test,
                "exit_code": 0,
                "passed": True,
                "stdout": "1 passed",
                "stderr": "",
                "duration_seconds": 0.1,
            }
        ]

        with (
            patch.object(WorkerLauncher, "_collect_diff", return_value="diff --git a/file"),
            patch.object(
                WorkerLauncher,
                "_run_verification_commands",
                new=AsyncMock(return_value=verification_results),
            ) as mock_verify,
        ):
            result = await launcher.wait("wo-1")

        assert result.exit_code == 0
        assert result.stdout == "output text"
        assert result.diff == "diff --git a/file"
        assert result.tests_run == [expected_test]
        assert result.verification_results == verification_results
        assert result.completed_at is not None
        mock_verify.assert_awaited_once_with("/tmp/wt", [expected_test])
        assert "wo-1" not in launcher._processes

    @pytest.mark.asyncio
    async def test_wait_attached_logs_stream_to_files(self, tmp_path: Path):
        launcher = WorkerLauncher(LaunchConfig(auto_commit=False))
        worktree = tmp_path / "wt"
        worktree.mkdir()

        worker = WorkerProcess(
            work_order_id="wo-live-logs",
            agent="codex",
            worktree_path=str(worktree),
            branch="main",
            pid=123,
        )
        launcher._workers["wo-live-logs"] = worker

        stdout_reader = asyncio.StreamReader()
        stdout_reader.feed_data(b"worker stdout\n")
        stdout_reader.feed_eof()

        stderr_reader = asyncio.StreamReader()
        stderr_reader.feed_data(b"worker stderr\n")
        stderr_reader.feed_eof()

        mock_proc = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0
        mock_proc.stdout = stdout_reader
        mock_proc.stderr = stderr_reader
        launcher._processes["wo-live-logs"] = mock_proc
        launcher._start_live_log_capture("wo-live-logs", str(worktree), mock_proc)

        with (
            patch.object(WorkerLauncher, "_collect_diff", return_value=""),
            patch.object(WorkerLauncher, "_git_output", return_value="abc123"),
            patch.object(WorkerLauncher, "_collect_commit_shas", return_value=[]),
            patch.object(WorkerLauncher, "_collect_changed_paths", return_value=[]),
        ):
            result = await launcher.wait("wo-live-logs")

        assert result.exit_code == 0
        assert result.stdout == "worker stdout\n"
        assert result.stderr == "worker stderr\n"
        assert (worktree / ".swarm_worker_stdout.log").read_text() == "worker stdout\n"
        assert (worktree / ".swarm_worker_stderr.log").read_text() == "worker stderr\n"

    @pytest.mark.asyncio
    async def test_wait_handles_timeout(self):
        launcher = WorkerLauncher(LaunchConfig(timeout_seconds=0.01, auto_commit=False))

        worker = WorkerProcess(
            work_order_id="wo-2",
            agent="codex",
            worktree_path="/tmp/wt",
            branch="main",
            pid=200,
        )
        launcher._workers["wo-2"] = worker

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        launcher._processes["wo-2"] = mock_proc

        with patch.object(WorkerLauncher, "_collect_diff", return_value=""):
            result = await launcher.wait("wo-2")

        assert result.exit_code == -1
        assert "Timed out" in result.stderr
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_salvages_codex_timeout_with_commit_and_runs_verification(self):
        launcher = WorkerLauncher(LaunchConfig(timeout_seconds=0.01, auto_commit=True))
        expected_test = "python -m pytest tests/swarm/test_worker_launcher.py -q"

        worker = WorkerProcess(
            work_order_id="wo-timeout-salvage",
            agent="codex",
            worktree_path="/tmp/wt",
            branch="main",
            pid=200,
            expected_tests=[expected_test],
        )
        launcher._workers["wo-timeout-salvage"] = worker

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        launcher._processes["wo-timeout-salvage"] = mock_proc

        verification_results = [
            {
                "command": expected_test,
                "exit_code": 0,
                "passed": True,
                "stdout": "1 passed",
                "stderr": "",
                "duration_seconds": 0.2,
            }
        ]

        with (
            patch.object(WorkerLauncher, "_collect_diff", return_value=""),
            patch.object(
                WorkerLauncher,
                "_has_working_tree_changes",
                new=AsyncMock(return_value=True),
            ),
            patch.object(WorkerLauncher, "_auto_commit", new=AsyncMock()) as mock_commit,
            patch.object(WorkerLauncher, "_git_output", return_value="abc123"),
            patch.object(WorkerLauncher, "_collect_commit_shas", return_value=["abc123"]),
            patch.object(WorkerLauncher, "_collect_changed_paths", return_value=["file.py"]),
            patch.object(
                WorkerLauncher,
                "_run_verification_commands",
                new=AsyncMock(return_value=verification_results),
            ) as mock_verify,
        ):
            result = await launcher.wait("wo-timeout-salvage")

        assert result.exit_code == -1
        assert "salvageable commit" not in result.stderr
        mock_proc.kill.assert_called_once()
        mock_commit.assert_not_awaited()
        mock_verify.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wait_unknown_raises(self):
        launcher = WorkerLauncher()
        with pytest.raises(KeyError, match="No running worker"):
            await launcher.wait("nonexistent")

    @pytest.mark.asyncio
    async def test_wait_runs_expected_verification_commands(self):
        launcher = WorkerLauncher(LaunchConfig(auto_commit=False))

        worker = WorkerProcess(
            work_order_id="wo-verify",
            agent="claude",
            worktree_path="/tmp/wt",
            branch="main",
            pid=100,
            expected_tests=["python -m pytest tests/swarm/test_worker_launcher.py -q"],
        )
        launcher._workers["wo-verify"] = worker

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"worker output", b""))
        mock_proc.returncode = 0
        launcher._processes["wo-verify"] = mock_proc

        verification_results = [
            {
                "command": "python -m pytest tests/swarm/test_worker_launcher.py -q",
                "exit_code": 0,
                "passed": True,
                "stdout": "1 passed",
                "stderr": "",
                "duration_seconds": 0.2,
            }
        ]

        with (
            patch.object(WorkerLauncher, "_collect_diff", return_value=""),
            patch.object(WorkerLauncher, "_git_output", return_value="abc123"),
            patch.object(WorkerLauncher, "_collect_commit_shas", return_value=["abc123"]),
            patch.object(WorkerLauncher, "_collect_changed_paths", return_value=["file.py"]),
            patch.object(
                WorkerLauncher,
                "_run_verification_commands",
                new=AsyncMock(return_value=verification_results),
            ),
        ):
            result = await launcher.wait("wo-verify")

        assert result.tests_run == ["python -m pytest tests/swarm/test_worker_launcher.py -q"]
        assert result.verification_results == verification_results


class TestLaunchAndWait:
    @pytest.mark.asyncio
    async def test_launch_and_wait_combined(self, tmp_path: Path):
        launcher = WorkerLauncher(LaunchConfig(auto_commit=False))
        wo = {"work_order_id": "wo-combo", "target_agent": "claude", "title": "Test"}
        worktree = tmp_path / "wt"
        (worktree / "scripts").mkdir(parents=True)
        (worktree / "scripts" / "codex_session.sh").write_text(
            "#!/usr/bin/env bash\n", encoding="utf-8"
        )

        mock_proc = AsyncMock()
        mock_proc.pid = 55
        mock_proc.communicate = AsyncMock(return_value=(b"done", b""))
        mock_proc.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch.object(WorkerLauncher, "_git_output", return_value=""),
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch.object(WorkerLauncher, "_collect_diff", return_value=""),
        ):
            result = await launcher.launch_and_wait(wo, worktree_path=str(worktree))

        assert result.exit_code == 0
        assert result.stdout == "done"


class TestVerificationCommands:
    def test_prepare_verification_command_wraps_pytest_with_pytest_main(self) -> None:
        prepared = WorkerLauncher._prepare_verification_command(
            "python -m pytest tests/swarm/test_supervisor.py -q --timeout=60 -x"
        )

        assert "import pytest" in prepared
        assert "pytest.main" in prepared
        assert "'tests/swarm/test_supervisor.py'" in prepared
        assert "'--timeout=60'" in prepared
        assert "'-x'" in prepared

    def test_verification_environment_adds_worktree_python_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        worktree = tmp_path / "wt"
        (worktree / "aragora-debate" / "src").mkdir(parents=True)
        monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")

        env = WorkerLauncher._verification_environment(str(worktree))

        pythonpath = env["PYTHONPATH"].split(os.pathsep)
        assert pythonpath[0] == str(worktree.resolve())
        assert str((worktree / "aragora-debate" / "src").resolve()) in pythonpath
        assert "/existing/pythonpath" in pythonpath

    def test_verification_environment_does_not_link_live_node_modules(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_ensure_live_node_modules is disabled -- no cross-worktree symlinks."""
        runtime_root = tmp_path / "runtime"
        source_node_modules = runtime_root / "aragora" / "live" / "node_modules"
        (source_node_modules / ".bin").mkdir(parents=True)
        worktree = tmp_path / "wt"
        (worktree / "aragora" / "live").mkdir(parents=True)

        monkeypatch.setattr(
            WorkerLauncher,
            "_runtime_repo_root",
            staticmethod(lambda: runtime_root),
        )

        env = WorkerLauncher._verification_environment(str(worktree))

        linked_node_modules = worktree / "aragora" / "live" / "node_modules"
        assert not linked_node_modules.exists()
        assert not linked_node_modules.is_symlink()
        assert "NODE_PATH" not in env or str(linked_node_modules) not in env["NODE_PATH"]

    @pytest.mark.asyncio
    async def test_run_verification_commands_uses_shared_verification_environment(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"ok\n", b""))
        mock_proc.returncode = 0

        with (
            patch.object(
                WorkerLauncher,
                "_verification_environment",
                return_value={"CUSTOM_ENV": "1"},
            ) as mock_env,
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
        ):
            result = await WorkerLauncher._run_verification_commands(
                "/tmp/wt",
                ["python -m pytest tests/swarm/test_supervisor.py -q"],
                timeout=30.0,
            )

        assert result[0]["passed"] is True
        mock_env.assert_called_once_with("/tmp/wt")
        assert mock_exec.await_args.kwargs["env"] == {"CUSTOM_ENV": "1"}

    @pytest.mark.asyncio
    async def test_run_verification_commands_wraps_pytest_commands(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            result = await WorkerLauncher._run_verification_commands(
                "/tmp/wt",
                ["python -m pytest tests/swarm/test_supervisor.py -q"],
                timeout=30.0,
            )

        assert result[0]["passed"] is True
        execution_command = mock_exec.await_args.args[2]
        assert "import pytest" in execution_command
        assert "pytest.main" in execution_command

    @pytest.mark.asyncio
    async def test_run_verification_commands_uses_current_python_interpreter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_python = tmp_path / "python"
        fake_python.write_text(
            "#!/bin/sh\necho 'broken python on PATH' >&2\nexit 126\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        script = tmp_path / "hello.py"
        script.write_text("print('ok from stable python')\n", encoding="utf-8")

        monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

        result = await WorkerLauncher._run_verification_commands(
            str(tmp_path),
            ["python hello.py"],
            timeout=30.0,
        )

        assert len(result) == 1
        assert result[0]["command"] == "python hello.py"
        assert result[0]["exit_code"] == 0
        assert result[0]["passed"] is True
        assert result[0]["stdout"].strip() == "ok from stable python"


class TestWaitDetached:
    @pytest.mark.asyncio
    async def test_wait_handles_none_stdout_stderr(self):
        """wait() should not crash when communicate() returns (None, None) in detached mode."""
        launcher = WorkerLauncher(LaunchConfig(auto_commit=False, detach=True))

        worker = WorkerProcess(
            work_order_id="wo-detach",
            agent="codex",
            worktree_path="/tmp/wt",
            branch="main",
            pid=100,
        )
        launcher._workers["wo-detach"] = worker

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(None, None))
        mock_proc.returncode = 0
        launcher._processes["wo-detach"] = mock_proc

        with (
            patch.object(WorkerLauncher, "_collect_diff", return_value=""),
            patch.object(WorkerLauncher, "_read_log_file", return_value="log output"),
        ):
            result = await launcher.wait("wo-detach")

        assert result.exit_code == 0
        assert result.stdout == "log output"
        assert result.stderr == "log output"
        assert result.completed_at is not None


class TestCollectFinishedSync:
    def test_collects_already_finished_in_memory_worker(self):
        launcher = WorkerLauncher(LaunchConfig(auto_commit=False))
        expected_test = "python -m pytest tests/swarm/test_supervisor.py -q"
        verification_results = [
            {
                "command": expected_test,
                "exit_code": 0,
                "passed": True,
                "stdout": "1 passed",
                "stderr": "",
                "duration_seconds": 0.1,
            }
        ]

        worker = WorkerProcess(
            work_order_id="wo-sync-finished",
            agent="codex",
            worktree_path="/tmp/wt",
            branch="main",
            pid=222,
            initial_head="def456",
            expected_tests=[expected_test],
        )
        launcher._workers["wo-sync-finished"] = worker
        proc = MagicMock()
        proc.returncode = 0
        launcher._processes["wo-sync-finished"] = proc

        with (
            patch.object(WorkerLauncher, "_collect_diff_sync", return_value="diff --git a/x"),
            patch.object(WorkerLauncher, "_git_output_sync", return_value="abc123"),
            patch.object(WorkerLauncher, "_read_log_file", return_value="some output"),
            patch.object(WorkerLauncher, "_collect_commit_shas_sync", return_value=["abc123"]),
            patch.object(WorkerLauncher, "_collect_changed_paths_sync", return_value=["file.py"]),
            patch.object(
                WorkerLauncher,
                "_run_verification_commands_sync",
                return_value=verification_results,
            ) as mock_verify,
            patch.object(WorkerLauncher, "_cleanup_session_artifacts"),
        ):
            completed = launcher.collect_finished_sync(work_order_ids=["wo-sync-finished"])

        assert len(completed) == 1
        result = completed[0]
        assert result.exit_code == 0
        assert result.head_sha == "abc123"
        assert result.commit_shas == ["abc123"]
        assert result.changed_paths == ["file.py"]
        assert result.tests_run == [expected_test]
        assert result.verification_results == verification_results
        assert result.stdout == "some output"
        mock_verify.assert_called_once_with("/tmp/wt", [expected_test])
        assert "wo-sync-finished" not in launcher._processes


class TestCollectDetachedResult:
    @pytest.mark.asyncio
    async def test_returns_none_if_pid_running(self):
        with patch.object(WorkerLauncher, "_is_pid_running", return_value=True):
            result = await WorkerLauncher.collect_detached_result(
                work_order_id="wo-1",
                agent="codex",
                worktree_path="/tmp/wt",
                branch="main",
                pid=12345,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_collects_results_when_pid_dead(self):
        expected_test = "python -m pytest tests/swarm/test_supervisor.py -q"
        verification_results = [
            {
                "command": expected_test,
                "exit_code": 0,
                "passed": True,
                "stdout": "1 passed",
                "stderr": "",
                "duration_seconds": 0.1,
            }
        ]
        with (
            patch.object(WorkerLauncher, "_is_pid_running", return_value=False),
            patch.object(WorkerLauncher, "_collect_diff", return_value="diff --git a/x"),
            patch.object(WorkerLauncher, "_auto_commit", new_callable=AsyncMock),
            patch.object(WorkerLauncher, "_git_output", return_value="abc123"),
            patch.object(WorkerLauncher, "_read_log_file", return_value="some output"),
            patch.object(WorkerLauncher, "_collect_commit_shas", return_value=["abc123"]),
            patch.object(WorkerLauncher, "_collect_changed_paths", return_value=["file.py"]),
            patch.object(
                WorkerLauncher,
                "_run_verification_commands",
                new=AsyncMock(return_value=verification_results),
            ) as mock_verify,
        ):
            result = await WorkerLauncher.collect_detached_result(
                work_order_id="wo-2",
                agent="codex",
                worktree_path="/tmp/wt",
                branch="main",
                pid=99999,
                initial_head="def456",
                expected_tests=[expected_test],
            )

        assert result is not None
        assert result.exit_code == 0
        assert result.head_sha == "abc123"
        assert result.commit_shas == ["abc123"]
        assert result.changed_paths == ["file.py"]
        assert result.tests_run == [expected_test]
        assert result.verification_results == verification_results
        assert result.stdout == "some output"
        mock_verify.assert_awaited_once_with("/tmp/wt", [expected_test])

    @pytest.mark.asyncio
    async def test_collects_without_pid(self):
        """When no PID is stored, always collect (assume finished)."""
        with (
            patch.object(WorkerLauncher, "_collect_diff", return_value=""),
            patch.object(WorkerLauncher, "_git_output", return_value="abc123"),
            patch.object(WorkerLauncher, "_read_log_file", return_value=""),
            patch.object(WorkerLauncher, "_collect_commit_shas", return_value=[]),
            patch.object(WorkerLauncher, "_collect_changed_paths", return_value=[]),
        ):
            result = await WorkerLauncher.collect_detached_result(
                work_order_id="wo-3",
                agent="claude",
                worktree_path="/tmp/wt",
                branch="main",
            )

        assert result is not None
        assert result.exit_code == 0


class TestSnapshotProgress:
    @pytest.mark.asyncio
    async def test_collects_git_state(self):
        launcher = WorkerLauncher()
        work_order = {
            "pid": 12345,
            "worktree_path": "/tmp/wt",
            "initial_head": "abc123",
        }

        with (
            patch.object(WorkerLauncher, "_is_pid_running", return_value=True),
            patch.object(WorkerLauncher, "_git_output", return_value="def456"),
            patch.object(WorkerLauncher, "_collect_diff", return_value="diff --git a/x\n+line\n"),
            patch.object(WorkerLauncher, "_collect_changed_paths", return_value=["file.py"]),
        ):
            snapshot = await launcher.snapshot_progress(work_order)

        assert snapshot["pid_alive"] is True
        assert snapshot["head_sha"] == "def456"
        assert snapshot["changed_paths"] == ["file.py"]
        assert snapshot["diff_lines"] == 2

    @pytest.mark.asyncio
    async def test_includes_log_tails(self, tmp_path: Path):
        launcher = WorkerLauncher()
        work_order = {
            "pid": 12345,
            "worktree_path": str(tmp_path),
            "initial_head": "abc123",
        }
        long_stdout = "a" * 5000
        long_stderr = "b" * 5000
        (tmp_path / ".swarm_worker_stdout.log").write_text(long_stdout, encoding="utf-8")
        (tmp_path / ".swarm_worker_stderr.log").write_text(long_stderr, encoding="utf-8")

        with (
            patch.object(WorkerLauncher, "_is_pid_running", return_value=True),
            patch.object(WorkerLauncher, "_git_output", return_value="def456"),
            patch.object(WorkerLauncher, "_collect_diff", return_value=""),
            patch.object(WorkerLauncher, "_collect_changed_paths", return_value=[]),
        ):
            snapshot = await launcher.snapshot_progress(work_order)

        assert snapshot["stdout_tail"] == long_stdout[-4000:]
        assert snapshot["stderr_tail"] == long_stderr[-4000:]


class TestCollectChangedPaths:
    @pytest.mark.asyncio
    async def test_skips_origin_main_fallback_when_initial_head_matches_head_sha(self):
        calls: list[tuple[str, ...]] = []

        async def _git_output(_worktree_path: str, *args: str) -> str:
            calls.append(tuple(args))
            if args[:2] == ("status", "--porcelain"):
                return ""
            return "aragora/swarm/boss_loop.py\n"

        with patch.object(WorkerLauncher, "_git_output", side_effect=_git_output):
            changed = await WorkerLauncher._collect_changed_paths(
                "/tmp/wt",
                initial_head="abc123",
                head_sha="abc123",
            )

        assert changed == []
        assert ("diff", "--name-only", "origin/main..HEAD") not in calls


class TestIsPidRunning:
    def test_current_process_is_running(self):
        import os

        assert WorkerLauncher._is_pid_running(os.getpid()) is True

    def test_nonexistent_pid(self):
        # PID 4000000 is very unlikely to exist
        assert WorkerLauncher._is_pid_running(4000000) is False


class TestReadLogFile:
    def test_reads_existing_log(self, tmp_path: Path):
        log_file = tmp_path / ".swarm_worker_stdout.log"
        log_file.write_text("hello world")
        assert WorkerLauncher._read_log_file(str(tmp_path), "stdout") == "hello world"

    def test_returns_empty_for_missing(self, tmp_path: Path):
        assert WorkerLauncher._read_log_file(str(tmp_path), "stdout") == ""


class TestActiveWorkers:
    def test_active_workers_list(self):
        launcher = WorkerLauncher()
        launcher._workers["a"] = WorkerProcess(
            work_order_id="a",
            agent="claude",
            worktree_path="/tmp",
            branch="main",
            pid=1,
        )
        launcher._workers["b"] = WorkerProcess(
            work_order_id="b",
            agent="codex",
            worktree_path="/tmp",
            branch="main",
            pid=2,
            exit_code=0,
        )
        active = launcher.active_workers()
        assert len(active) == 1
        assert active[0].work_order_id == "a"


class TestEnsureLiveNodeModulesDisabled:
    def test_always_returns_none(self, tmp_path: Path) -> None:
        """_ensure_live_node_modules must never create cross-worktree symlinks."""
        worktree = tmp_path / "wt"
        (worktree / "aragora" / "live").mkdir(parents=True)
        result = WorkerLauncher._ensure_live_node_modules(worktree)
        assert result is None
        assert not (worktree / "aragora" / "live" / "node_modules").exists()
