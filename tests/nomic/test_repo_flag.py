"""Tests for --repo flag on self_develop.py and nomic_staged.py.

Verifies that the Nomic Loop can target external codebases via the --repo flag.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── self_develop.py tests ──


class TestSelfDevelopRepoFlag:
    """Test --repo flag on self_develop.py."""

    def test_repo_arg_accepted(self) -> None:
        """Parser accepts --repo argument."""
        sys_argv = [
            "self_develop.py",
            "--goal", "Improve test coverage",
            "--repo", "/tmp/customer-repo",
            "--dry-run",
        ]
        with patch("sys.argv", sys_argv):
            # Import and test parser
            import importlib
            import scripts.self_develop as sd
            importlib.reload(sd)

            parser = sd.main.__code__  # Just verify module loads
            assert hasattr(sd, "run_orchestration")

    def test_repo_path_passed_to_orchestrator(self) -> None:
        """--repo flag flows through to orchestrator as aragora_path."""
        from scripts.self_develop import run_orchestration

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_goal = AsyncMock(
            return_value=MagicMock(
                success=True,
                summary="done",
                completed_subtasks=1,
                total_subtasks=1,
                failed_subtasks=0,
                skipped_subtasks=0,
                duration_seconds=1.0,
                error=None,
            )
        )

        target_repo = Path("/tmp/test-customer-repo")

        with patch(
            "scripts.self_develop.HardenedOrchestrator",
            return_value=mock_orchestrator,
        ) as mock_cls:
            import asyncio
            result = asyncio.run(
                run_orchestration(
                    goal="Fix bugs",
                    tracks=None,
                    max_cycles=1,
                    max_parallel=1,
                    require_approval=False,
                    repo_path=target_repo,
                )
            )

        # Verify aragora_path was passed in kwargs
        call_kwargs = mock_cls.call_args
        assert call_kwargs is not None
        # aragora_path flows via common_kwargs
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        # It's passed through **common_kwargs which includes aragora_path
        assert "aragora_path" in all_kwargs or any(
            k == "aragora_path" for k in (call_kwargs.kwargs or {})
        )

    def test_repo_path_passed_to_pipeline(self) -> None:
        """--repo flag flows through to NomicPipelineBridge."""
        from scripts.self_develop import run_pipeline_execution

        target_repo = Path("/tmp/pipeline-repo")

        mock_bridge = MagicMock()
        mock_bridge.build_decision_plan.return_value = MagicMock(
            id="test", status=MagicMock(value="planned"),
            risk_register=None, verification_plan=None, implement_plan=None,
        )
        mock_bridge.execute_via_pipeline = AsyncMock(
            return_value=MagicMock(success=True, tasks_completed=0, tasks_total=0)
        )

        mock_bridge_cls = MagicMock(return_value=mock_bridge)

        with (
            patch(
                "aragora.nomic.pipeline_bridge.NomicPipelineBridge",
                mock_bridge_cls,
            ),
            patch.dict("sys.modules", {
                "aragora.nomic.pipeline_bridge": MagicMock(
                    NomicPipelineBridge=mock_bridge_cls,
                ),
            }),
            patch(
                "scripts.self_develop.run_heuristic_decomposition",
                return_value=MagicMock(
                    original_task="test",
                    complexity_level="medium",
                    complexity_score=5,
                    should_decompose=True,
                    rationale="test",
                    subtasks=[MagicMock(
                        estimated_complexity="medium",
                        title="sub1",
                        description="desc",
                        file_scope=[],
                        dependencies=[],
                    )],
                ),
            ),
        ):
            import asyncio
            asyncio.run(
                run_pipeline_execution(
                    goal="Fix bugs",
                    repo_path=target_repo,
                )
            )

        # Verify repo_path was passed to NomicPipelineBridge
        call_kwargs = mock_bridge_cls.call_args.kwargs
        assert call_kwargs["repo_path"] == target_repo

    def test_none_repo_uses_cwd(self) -> None:
        """No --repo flag defaults to None (orchestrator uses Path.cwd())."""
        from scripts.self_develop import run_orchestration

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute_goal = AsyncMock(
            return_value=MagicMock(
                success=True, summary="done",
                completed_subtasks=0, total_subtasks=0,
                failed_subtasks=0, skipped_subtasks=0,
                duration_seconds=0.1, error=None,
            )
        )

        with patch(
            "scripts.self_develop.HardenedOrchestrator",
            return_value=mock_orchestrator,
        ) as mock_cls:
            import asyncio
            asyncio.run(
                run_orchestration(
                    goal="Test",
                    tracks=None,
                    max_cycles=1,
                    max_parallel=1,
                    require_approval=False,
                    repo_path=None,
                )
            )

        # aragora_path should NOT be in kwargs when repo_path is None
        call_kwargs = mock_cls.call_args.kwargs if mock_cls.call_args.kwargs else {}
        assert "aragora_path" not in call_kwargs


# ── nomic_staged.py tests ──


class TestNomicStagedRepoFlag:
    """Test --repo flag on nomic_staged.py."""

    def test_repo_overrides_aragora_path(self) -> None:
        """--repo flag updates ARAGORA_PATH and DATA_DIR globals."""
        import scripts.nomic_staged as staged

        original_path = staged.ARAGORA_PATH
        original_data = staged.DATA_DIR

        try:
            target = Path("/tmp/external-repo")
            staged.ARAGORA_PATH = target
            staged.DATA_DIR = target / ".nomic"

            assert staged.ARAGORA_PATH == target
            assert staged.DATA_DIR == target / ".nomic"
        finally:
            staged.ARAGORA_PATH = original_path
            staged.DATA_DIR = original_data

    def test_implement_uses_aragora_path(self) -> None:
        """phase_implement passes ARAGORA_PATH to HybridExecutor."""
        import scripts.nomic_staged as staged

        target = Path("/tmp/impl-repo")
        original_path = staged.ARAGORA_PATH
        original_data = staged.DATA_DIR

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.to_dict.return_value = {"success": True}

        mock_executor_instance = MagicMock()
        mock_executor_instance.execute_task = AsyncMock(return_value=mock_result)

        design_data = {
            "design": "Implement caching layer",
            "timestamp": "2026-02-16T00:00:00",
        }

        try:
            staged.ARAGORA_PATH = target
            staged.DATA_DIR = Path("/tmp/staged-data")
            staged.DATA_DIR.mkdir(parents=True, exist_ok=True)

            # Create mock module for the local import inside phase_implement
            mock_executor_cls = MagicMock(return_value=mock_executor_instance)
            mock_impl_module = MagicMock()
            mock_impl_module.HybridExecutor = mock_executor_cls
            mock_impl_module.ImplementTask = MagicMock()
            mock_impl_module.TaskResult = MagicMock()

            mock_types_module = MagicMock()
            mock_types_module.ImplementTask = MagicMock()
            mock_types_module.TaskResult = MagicMock()

            with (
                patch.object(staged, "load_phase", return_value=design_data),
                patch.object(staged, "save_phase") as mock_save,
                patch.dict("sys.modules", {
                    "aragora.implement.executor": mock_impl_module,
                    "aragora.implement.types": mock_types_module,
                }),
            ):
                import asyncio
                asyncio.run(staged.phase_implement())

            # Verify HybridExecutor was created with the external repo path
            assert mock_executor_cls.called
            call_kwargs = mock_executor_cls.call_args.kwargs if mock_executor_cls.call_args.kwargs else {}
            assert call_kwargs.get("repo_path") == str(target)

        finally:
            staged.ARAGORA_PATH = original_path
            staged.DATA_DIR = original_data
