"""Tests for WorkerLauncher — spawns and monitors CLI worker processes."""

from __future__ import annotations

import asyncio
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

        assert "Codex lane discipline:" in prompt
        assert "checkpoint commit before long" in prompt
        assert "Do not exit 0 with staged or unstaged changes remaining." in prompt

    def test_claude_prompt_omits_codex_lane_closure_guidance(self):
        prompt = WorkerLauncher._build_prompt(
            {
                "target_agent": "claude",
                "title": "Harden execution",
            }
        )

        assert "Codex lane discipline:" not in prompt


class TestBuildCommand:
    def test_claude_command(self):
        launcher = WorkerLauncher(LaunchConfig(claude_model="claude-opus-4-6"))
        cmd = launcher._build_command("claude", "fix bug", "/tmp/wt")
        assert cmd[0] == "bash"
        assert "scripts/codex_session.sh" in cmd[1]
        assert "--session-id" not in cmd
        assert "--" in cmd
        assert "-p" in cmd
        assert "fix bug" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--model" in cmd
        assert "claude-opus-4-6" in cmd

    def test_codex_command(self):
        launcher = WorkerLauncher(LaunchConfig(codex_model="o3"))
        cmd = launcher._build_command("codex", "fix bug", "/tmp/wt")
        assert cmd[0] == "bash"
        assert "exec" in cmd
        assert "fix bug" in cmd
        assert "--full-auto" in cmd
        assert "--model" in cmd
        assert "o3" in cmd

    def test_unknown_agent_falls_back_to_claude(self):
        launcher = WorkerLauncher()
        cmd = launcher._build_command("gpt5", "do thing", "/tmp/wt")
        assert cmd[0] == "bash"
        assert "--dangerously-skip-permissions" in cmd

    def test_no_model_flag_when_none(self):
        launcher = WorkerLauncher()
        cmd = launcher._build_command("claude", "task", "/tmp/wt")
        assert "--model" not in cmd

    def test_direct_cli_command_when_session_wrapper_disabled(self):
        launcher = WorkerLauncher(LaunchConfig(use_managed_session_script=False))
        cmd = launcher._build_command("claude", "task", "/tmp/wt")
        assert cmd[0] == "claude"
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
    async def test_launch_raises_on_missing_cli(self):
        launcher = WorkerLauncher()
        wo = {"work_order_id": "wo-1", "target_agent": "claude"}

        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="CLI not found"):
                await launcher.launch(wo, worktree_path="/tmp/wt")


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

        assert result.exit_code == 0
        assert result.commit_shas == ["abc123"]
        assert "salvageable commit" in result.stderr
        mock_proc.kill.assert_called_once()
        mock_commit.assert_awaited_once()
        mock_verify.assert_awaited_once_with("/tmp/wt", [expected_test])

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
