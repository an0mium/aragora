"""Tests for sandbox integration in agent execution.

Verifies that:
- build_worktree_docker_args generates correct Docker CLI arguments
- HybridExecutor sandbox_mode routes to Docker execution
- ImplementPhase propagates sandbox_mode to executor
- Sandbox execution handles errors gracefully
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.sandbox.executor import build_worktree_docker_args


class TestBuildWorktreeDockerArgs:
    """Tests for build_worktree_docker_args helper."""

    def test_basic_args(self):
        """Should produce valid docker run arguments."""
        args = build_worktree_docker_args(
            worktree_path=Path("/tmp/worktree"),
        )
        assert "--rm" in args
        assert "--security-opt=no-new-privileges" in args
        assert "-v" in args
        assert "/tmp/worktree:/workspace:rw" in args
        assert "-w" in args
        assert "/workspace" in args
        assert "python:3.11-slim" in args

    def test_network_disabled_by_default(self):
        """Network should be disabled by default."""
        args = build_worktree_docker_args(
            worktree_path=Path("/tmp/wt"),
        )
        assert "--network=none" in args

    def test_network_enabled(self):
        """Network can be enabled."""
        args = build_worktree_docker_args(
            worktree_path=Path("/tmp/wt"),
            network=True,
        )
        assert "--network=none" not in args

    def test_custom_image(self):
        """Custom Docker image should be used."""
        args = build_worktree_docker_args(
            worktree_path=Path("/tmp/wt"),
            image="node:18-slim",
        )
        assert "node:18-slim" in args

    def test_custom_memory(self):
        """Custom memory limit should be set."""
        args = build_worktree_docker_args(
            worktree_path=Path("/tmp/wt"),
            memory_mb=4096,
        )
        assert "--memory=4096m" in args

    def test_repo_root_mounted_readonly(self):
        """Repo root should be mounted read-only when different from worktree."""
        args = build_worktree_docker_args(
            worktree_path=Path("/tmp/worktree"),
            repo_root=Path("/home/user/repo"),
        )
        assert "/home/user/repo:/repo:ro" in " ".join(args)
        assert "/tmp/worktree:/workspace:rw" in " ".join(args)

    def test_repo_root_same_as_worktree_not_duplicated(self, tmp_path):
        """When repo_root equals worktree, should not duplicate mount."""
        wt = tmp_path / "wt"
        wt.mkdir()
        args = build_worktree_docker_args(
            worktree_path=wt,
            repo_root=wt,
        )
        # Should only have one -v mount (the workspace)
        v_count = args.count("-v")
        assert v_count == 1

    def test_pids_limit_present(self):
        """Should set pids limit for safety."""
        args = build_worktree_docker_args(
            worktree_path=Path("/tmp/wt"),
        )
        assert "--pids-limit" in args


class TestHybridExecutorSandboxMode:
    """Tests for HybridExecutor with sandbox_mode."""

    def test_sandbox_mode_default_on(self):
        """sandbox_mode should default to True for security."""
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(repo_path=Path("/tmp/test"))
        assert executor.sandbox_mode is True

    def test_sandbox_mode_on(self):
        """sandbox_mode should be configurable."""
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(
            repo_path=Path("/tmp/test"),
            sandbox_mode=True,
        )
        assert executor.sandbox_mode is True

    def test_sandbox_image_configurable(self):
        """Sandbox image should be configurable."""
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(
            repo_path=Path("/tmp/test"),
            sandbox_mode=True,
            sandbox_image="node:18-slim",
        )
        assert executor.sandbox_image == "node:18-slim"

    def test_sandbox_memory_configurable(self):
        """Sandbox memory should be configurable."""
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(
            repo_path=Path("/tmp/test"),
            sandbox_mode=True,
            sandbox_memory_mb=4096,
        )
        assert executor.sandbox_memory_mb == 4096

    @pytest.mark.asyncio
    async def test_sandbox_mode_routes_to_docker(self, tmp_path):
        """When sandbox_mode=True and use_harness=False, first attempt should use Docker sandbox."""
        from aragora.implement.executor import HybridExecutor
        from aragora.implement.types import ImplementTask

        executor = HybridExecutor(
            repo_path=tmp_path,
            sandbox_mode=True,
            use_harness=False,
        )

        task = ImplementTask(
            id="task-1",
            description="Add a feature",
            files=["aragora/server/handler.py"],
            complexity="simple",
        )

        with patch.object(executor, "_execute_in_sandbox", new_callable=AsyncMock) as mock_sandbox:
            mock_sandbox.return_value = MagicMock(
                task_id="task-1", success=True, diff="", model_used="sandbox:claude"
            )
            result = await executor.execute_task(task)

        mock_sandbox.assert_called_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_sandbox_mode_skipped_on_fallback(self, tmp_path):
        """Fallback attempts should not use sandbox (uses Codex directly)."""
        from aragora.implement.executor import HybridExecutor
        from aragora.implement.types import ImplementTask

        executor = HybridExecutor(
            repo_path=tmp_path,
            sandbox_mode=True,
        )

        task = ImplementTask(
            id="task-1",
            description="Add a feature",
            files=["aragora/server/handler.py"],
            complexity="simple",
        )

        mock_agent = MagicMock()
        mock_agent.generate = AsyncMock(return_value="done")
        mock_agent.name = "codex-fallback"
        mock_agent.timeout = 300

        with patch.object(executor, "_execute_in_sandbox", new_callable=AsyncMock) as mock_sandbox:
            with patch.object(executor, "_select_agent", return_value=(mock_agent, "codex")):
                with patch.object(executor, "_get_git_diff", return_value=""):
                    with patch(
                        "aragora.server.stream.arena_hooks.streaming_task_context",
                        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
                    ):
                        result = await executor.execute_task(task, use_fallback=True)

        mock_sandbox.assert_not_called()

    @pytest.mark.asyncio
    async def test_sandbox_mode_skipped_on_retry(self, tmp_path):
        """Retry attempts (attempt > 1) should not use sandbox."""
        from aragora.implement.executor import HybridExecutor
        from aragora.implement.types import ImplementTask

        executor = HybridExecutor(
            repo_path=tmp_path,
            sandbox_mode=True,
        )

        task = ImplementTask(
            id="task-1",
            description="Add a feature",
            files=[],
            complexity="simple",
        )

        mock_agent = MagicMock()
        mock_agent.generate = AsyncMock(return_value="done")
        mock_agent.name = "claude-implementer"
        mock_agent.timeout = 300

        with patch.object(executor, "_execute_in_sandbox", new_callable=AsyncMock) as mock_sandbox:
            with patch.object(executor, "_select_agent", return_value=(mock_agent, "claude")):
                with patch.object(executor, "_get_git_diff", return_value=""):
                    with patch(
                        "aragora.server.stream.arena_hooks.streaming_task_context",
                        return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
                    ):
                        result = await executor.execute_task(task, attempt=2)

        mock_sandbox.assert_not_called()

    def test_get_sandbox_docker_args(self):
        """_get_sandbox_docker_args should return valid Docker args."""
        from aragora.implement.executor import HybridExecutor

        executor = HybridExecutor(
            repo_path=Path("/tmp/test-repo"),
            sandbox_mode=True,
            sandbox_image="python:3.12-slim",
            sandbox_memory_mb=1024,
        )

        args = executor._get_sandbox_docker_args()
        joined = " ".join(args)
        assert "/tmp/test-repo:/workspace:rw" in joined
        assert "python:3.12-slim" in joined
        assert "--memory=1024m" in joined
        # Network enabled for LLM API calls
        assert "--network=none" not in joined


class TestImplementPhaseSandboxPropagation:
    """Tests for ImplementPhase sandbox_mode propagation."""

    def test_sandbox_mode_stored(self):
        """ImplementPhase should store sandbox_mode."""
        from scripts.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=Path("/tmp/test"),
            sandbox_mode=True,
        )
        assert phase.sandbox_mode is True

    def test_sandbox_mode_default_false(self):
        """sandbox_mode should default to False."""
        from scripts.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=Path("/tmp/test"),
        )
        assert phase.sandbox_mode is False

    @pytest.mark.asyncio
    async def test_sandbox_propagated_to_executor(self):
        """sandbox_mode should be propagated to executor during execute()."""
        from scripts.nomic.phases.implement import ImplementPhase

        mock_executor = MagicMock()
        mock_executor.sandbox_mode = False
        mock_executor.execute_plan = AsyncMock(return_value=[])

        mock_plan = MagicMock()
        mock_plan.tasks = []
        mock_plan.design_hash = "abc"

        phase = ImplementPhase(
            aragora_path=Path("/tmp/test"),
            executor=mock_executor,
            plan_generator=AsyncMock(return_value=mock_plan),
            sandbox_mode=True,
            log_fn=lambda msg: None,
            stream_emit_fn=lambda *args: None,
            record_replay_fn=lambda *args: None,
            save_state_fn=lambda state: None,
        )

        with patch.object(phase, "_git_stash_create", new_callable=AsyncMock, return_value=None):
            with patch.object(phase, "_get_git_diff", new_callable=AsyncMock, return_value=""):
                with patch.object(
                    phase, "_get_modified_files", new_callable=AsyncMock, return_value=[]
                ):
                    await phase.execute("design text")

        assert mock_executor.sandbox_mode is True
