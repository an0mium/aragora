"""Tests for nomic_staged.py phase_implement() → HybridExecutor wiring.

Validates that the staged execution path actually invokes code generation
rather than just prompting the user to implement manually.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.implement.types import ImplementTask, TaskResult


@pytest.fixture
def nomic_data_dir(tmp_path):
    """Temporary .nomic data directory with a design phase result."""
    data_dir = tmp_path / ".nomic"
    data_dir.mkdir()

    design_data = {
        "timestamp": "2026-02-16T12:00:00",
        "design": "Add retry logic to the HTTP connector:\n"
        "1. Create a RetryConfig dataclass\n"
        "2. Add exponential backoff to fetch()\n"
        "3. Add max_retries parameter",
        "consensus_reached": True,
    }
    (data_dir / "design.json").write_text(json.dumps(design_data))
    return data_dir


def _make_successful_result() -> TaskResult:
    return TaskResult(
        task_id="nomic-staged-001",
        success=True,
        diff="diff --git a/aragora/connectors/http.py\n+    max_retries: int = 3\n",
        model_used="claude",
        duration_seconds=12.5,
    )


def _make_failed_result() -> TaskResult:
    return TaskResult(
        task_id="nomic-staged-001",
        success=False,
        error="Claude agent timed out",
        model_used="claude",
        duration_seconds=60.0,
    )


class TestStagedImplementPhase:
    """Test that phase_implement() invokes HybridExecutor."""

    @pytest.mark.asyncio
    async def test_invokes_hybrid_executor(self, nomic_data_dir):
        """phase_implement() should create an ImplementTask and call executor."""
        import scripts.nomic_staged as staged

        staged.DATA_DIR = nomic_data_dir

        mock_executor = MagicMock()
        mock_executor.execute_task = AsyncMock(return_value=_make_successful_result())

        with patch.object(staged, "ARAGORA_PATH", nomic_data_dir.parent):
            with patch(
                "aragora.implement.executor.HybridExecutor",
                return_value=mock_executor,
            ):
                result = await staged.phase_implement()

        assert result["status"] == "implemented"
        assert result["task_result"]["success"] is True
        assert result["task_result"]["task_id"] == "nomic-staged-001"

        # Verify executor was called with an ImplementTask
        mock_executor.execute_task.assert_called_once()
        task_arg = mock_executor.execute_task.call_args[0][0]
        assert isinstance(task_arg, ImplementTask)
        assert task_arg.id == "nomic-staged-001"
        assert "retry" in task_arg.description.lower()
        assert task_arg.complexity == "complex"

    @pytest.mark.asyncio
    async def test_saves_implement_phase_data(self, nomic_data_dir):
        """phase_implement() should save results to implement.json."""
        import scripts.nomic_staged as staged

        staged.DATA_DIR = nomic_data_dir

        mock_executor = MagicMock()
        mock_executor.execute_task = AsyncMock(return_value=_make_successful_result())

        with patch.object(staged, "ARAGORA_PATH", nomic_data_dir.parent):
            with patch(
                "aragora.implement.executor.HybridExecutor",
                return_value=mock_executor,
            ):
                await staged.phase_implement()

        # Check the saved file
        impl_path = nomic_data_dir / "implement.json"
        assert impl_path.exists()
        saved = json.loads(impl_path.read_text())
        assert saved["status"] == "implemented"
        assert saved["task_result"]["success"] is True
        assert saved["task_result"]["diff"] != ""

    @pytest.mark.asyncio
    async def test_handles_executor_failure(self, nomic_data_dir):
        """phase_implement() should handle executor failures gracefully."""
        mock_executor = MagicMock()
        mock_executor.execute_task = AsyncMock(return_value=_make_failed_result())

        import scripts.nomic_staged as staged

        staged.DATA_DIR = nomic_data_dir

        with patch.object(staged, "ARAGORA_PATH", nomic_data_dir.parent):
            with patch(
                "aragora.implement.executor.HybridExecutor",
                return_value=mock_executor,
            ):
                result = await staged.phase_implement()

        assert result["status"] == "failed"
        assert result["task_result"]["success"] is False
        assert result["task_result"]["error"] == "Claude agent timed out"

    @pytest.mark.asyncio
    async def test_falls_back_to_manual_on_import_error(self, nomic_data_dir):
        """phase_implement() should fall back to manual mode if executor unavailable."""
        import scripts.nomic_staged as staged

        staged.DATA_DIR = nomic_data_dir

        with patch.object(staged, "ARAGORA_PATH", nomic_data_dir.parent):
            # Force ImportError by patching the import target
            with patch(
                "aragora.implement.executor.HybridExecutor",
                side_effect=ImportError("No module"),
            ):
                # Since the import is inside a try block, we need to trigger
                # the ImportError at import time. Patch builtins.__import__
                original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

                def selective_import(name, *args, **kwargs):
                    if name == "aragora.implement.executor":
                        raise ImportError("No module")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=selective_import):
                    result = await staged.phase_implement()

        assert result["status"] == "ready_for_implementation"
        assert "instructions" in result

    @pytest.mark.asyncio
    async def test_design_flows_into_implement(self, nomic_data_dir):
        """Design output should be used as the implementation task description."""
        import scripts.nomic_staged as staged

        staged.DATA_DIR = nomic_data_dir

        mock_executor = MagicMock()
        mock_executor.execute_task = AsyncMock(return_value=_make_successful_result())

        with patch.object(staged, "ARAGORA_PATH", nomic_data_dir.parent):
            with patch(
                "aragora.implement.executor.HybridExecutor",
                return_value=mock_executor,
            ):
                result = await staged.phase_implement()

        # The task description should contain the design
        task_arg = mock_executor.execute_task.call_args[0][0]
        assert "RetryConfig" in task_arg.description
        assert "exponential backoff" in task_arg.description

    @pytest.mark.asyncio
    async def test_executor_receives_repo_path(self, nomic_data_dir):
        """HybridExecutor should be initialized with the correct repo path."""
        import scripts.nomic_staged as staged

        staged.DATA_DIR = nomic_data_dir

        mock_executor = MagicMock()
        mock_executor.execute_task = AsyncMock(return_value=_make_successful_result())

        with patch.object(staged, "ARAGORA_PATH", nomic_data_dir.parent):
            with patch(
                "aragora.implement.executor.HybridExecutor",
                return_value=mock_executor,
            ) as mock_cls:
                await staged.phase_implement()

        # Verify the executor was created with repo_path and memory_gateway
        from unittest.mock import ANY
        mock_cls.assert_called_once_with(repo_path=str(nomic_data_dir.parent), memory_gateway=ANY)


class TestManualFallback:
    """Test the _phase_implement_manual fallback path."""

    def test_manual_fallback_returns_ready_status(self, nomic_data_dir):
        """Manual fallback should return ready_for_implementation status."""
        import scripts.nomic_staged as staged

        staged.DATA_DIR = nomic_data_dir

        result = staged._phase_implement_manual("Test design")
        assert result["status"] == "ready_for_implementation"
        assert result["design"] == "Test design"

    def test_manual_fallback_saves_phase(self, nomic_data_dir):
        """Manual fallback should save implement.json."""
        import scripts.nomic_staged as staged

        staged.DATA_DIR = nomic_data_dir

        staged._phase_implement_manual("Test design")
        impl_path = nomic_data_dir / "implement.json"
        assert impl_path.exists()
