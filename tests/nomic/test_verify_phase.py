"""
Tests for Nomic Loop Verify Phase.

Phase 4: Verification
- Tests test execution
- Tests type checking
- Tests lint checking
- Tests safety gates
- Tests rollback on failure
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


class TestVerifyPhaseExecution:
    """Tests for verify phase execution."""

    @pytest.mark.asyncio
    async def test_runs_pytest_successfully(self, mock_aragora_path, mock_verification_result):
        """Should run pytest and return results."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"All tests passed", b""))
            mock_exec.return_value = mock_proc

            result = await phase._run_tests()

            assert result["passed"] is True
            mock_exec.assert_called()

    @pytest.mark.asyncio
    async def test_detects_test_failures(self, mock_aragora_path):
        """Should detect and report test failures."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"FAILED test_example.py::test_func", b""))
            mock_exec.return_value = mock_proc

            result = await phase._run_tests()

            assert result["passed"] is False
            assert result["check"] == "tests"

    @pytest.mark.asyncio
    async def test_runs_syntax_checking(self, mock_aragora_path):
        """Should run syntax checking via AST."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc

            result = await phase._check_syntax()

            assert result["check"] == "syntax"

    @pytest.mark.asyncio
    async def test_detects_syntax_errors(self, mock_aragora_path):
        """Should detect syntax errors."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"SyntaxError: invalid syntax", b""))
            mock_exec.return_value = mock_proc

            result = await phase._check_syntax()

            assert result["check"] == "syntax"
            assert result["passed"] is False


class TestVerifyPhaseSafetyGates:
    """Tests for safety gate checks.

    Note: Safety gates are implemented in the nomic loop orchestrator,
    not in the VerifyPhase directly. These tests verify the expected
    behavior of the changed files detection.
    """

    @pytest.mark.asyncio
    async def test_detects_changed_files(self, mock_aragora_path):
        """Should detect changed files via git."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"aragora/core.py\nCLAUDE.md", b""))
            mock_exec.return_value = mock_proc

            changed = await phase._get_changed_files()

            assert "aragora/core.py" in changed
            assert "CLAUDE.md" in changed

    @pytest.mark.asyncio
    async def test_handles_no_changes(self, mock_aragora_path):
        """Should handle case with no changes."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc

            changed = await phase._get_changed_files()

            assert changed == []

    @pytest.mark.asyncio
    async def test_handles_git_error(self, mock_aragora_path):
        """Should handle git command errors gracefully."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 128
            mock_proc.communicate = AsyncMock(return_value=(b"", b"fatal: not a git repository"))
            mock_exec.return_value = mock_proc

            changed = await phase._get_changed_files()

            # Should return empty list on error
            assert changed == []


class TestVerifyPhaseRollback:
    """Tests for rollback functionality.

    Note: Rollback is handled by the RollbackManager in the nomic loop,
    not by VerifyPhase directly. These tests verify execute() behavior.
    """

    @pytest.mark.asyncio
    async def test_execute_returns_result(self, mock_aragora_path):
        """Should return VerifyResult from execute."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {"check": "tests", "passed": True}

                    result = await phase.execute()

                    assert result is not None
                    assert "success" in result

    @pytest.mark.asyncio
    async def test_execute_fails_on_test_failure(self, mock_aragora_path):
        """Should fail verification when tests fail."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {"check": "tests", "passed": False, "error": "Test failed"}

                    result = await phase.execute()

                    assert result["success"] is False



class TestVerifyPhaseIntegration:
    """Integration tests for verify phase."""

    @pytest.mark.asyncio
    async def test_full_verification_flow_success(
        self, mock_aragora_path, mock_verification_result
    ):
        """Should complete full verification flow on success."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {"check": "tests", "passed": True}

                    result = await phase.execute()

                    assert result["success"] is True
                    mock_tests.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_verification_flow_failure(self, mock_aragora_path):
        """Should report failure on any verification step failing."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {
                        "check": "tests",
                        "passed": False,
                        "output": "FAILED test_example.py::test_func",
                    }

                    result = await phase.execute()

                    assert result["success"] is False
