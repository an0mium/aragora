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

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="All tests passed",
                stderr="",
            )

            result = await phase.run_tests()

            assert result["passed"] is True
            mock_run.assert_called()

    @pytest.mark.asyncio
    async def test_detects_test_failures(self, mock_aragora_path):
        """Should detect and report test failures."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="FAILED test_example.py::test_func",
                stderr="AssertionError",
            )

            result = await phase.run_tests()

            assert result["passed"] is False
            assert "failures" in result

    @pytest.mark.asyncio
    async def test_runs_mypy_type_checking(self, mock_aragora_path):
        """Should run mypy type checking."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Success: no issues found",
                stderr="",
            )

            result = await phase.run_type_check()

            assert result["clean"] is True

    @pytest.mark.asyncio
    async def test_detects_mypy_errors(self, mock_aragora_path):
        """Should detect mypy type errors."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="aragora/core.py:10: error: Incompatible types",
                stderr="",
            )

            result = await phase.run_type_check()

            assert result["clean"] is False
            assert "errors" in result


class TestVerifyPhaseSafetyGates:
    """Tests for safety gate checks."""

    @pytest.mark.asyncio
    async def test_blocks_on_protected_file_changes(self, mock_aragora_path):
        """Should block changes to protected files."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
            protected_files=["CLAUDE.md", "core.py"],
        )

        changes = {
            "files_modified": ["aragora/core.py", "CLAUDE.md"],
        }

        result = await phase.check_safety_gates(changes)

        assert result["safe"] is False
        assert "protected" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_allows_non_protected_file_changes(self, mock_aragora_path):
        """Should allow changes to non-protected files."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
            protected_files=["CLAUDE.md"],
        )

        changes = {
            "files_modified": ["aragora/utils.py", "tests/test_utils.py"],
        }

        result = await phase.check_safety_gates(changes)

        assert result["safe"] is True

    @pytest.mark.asyncio
    async def test_blocks_on_excessive_changes(self, mock_aragora_path):
        """Should block excessively large changes."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
            max_files_changed=5,
        )

        changes = {
            "files_modified": [f"file{i}.py" for i in range(20)],
        }

        result = await phase.check_safety_gates(changes)

        assert result["safe"] is False
        assert "too many" in result["reason"].lower()


class TestVerifyPhaseRollback:
    """Tests for rollback functionality."""

    @pytest.mark.asyncio
    async def test_creates_backup_before_changes(self, mock_aragora_path):
        """Should create backup before applying changes."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch("shutil.copytree") as mock_copy:
            backup_path = await phase.create_backup()

            mock_copy.assert_called_once()
            assert backup_path is not None

    @pytest.mark.asyncio
    async def test_restores_from_backup_on_failure(self, mock_aragora_path):
        """Should restore from backup when verification fails."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        backup_path = mock_aragora_path / ".backup"
        backup_path.mkdir()

        with patch("shutil.rmtree") as mock_rm:
            with patch("shutil.copytree") as mock_copy:
                await phase.restore_from_backup(backup_path)

                # Should remove current state and restore backup
                assert mock_copy.called or mock_rm.called


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

        with patch.object(phase, "run_tests", new_callable=AsyncMock) as mock_tests:
            with patch.object(phase, "run_type_check", new_callable=AsyncMock) as mock_mypy:
                with patch.object(phase, "run_lint", new_callable=AsyncMock) as mock_lint:
                    mock_tests.return_value = {"passed": True, "failures": []}
                    mock_mypy.return_value = {"clean": True, "errors": []}
                    mock_lint.return_value = {"clean": True, "issues": []}

                    result = await phase.run()

                    assert result["passed"] is True
                    mock_tests.assert_called_once()
                    mock_mypy.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_verification_flow_failure(self, mock_aragora_path):
        """Should report failure on any verification step failing."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(),
        )

        with patch.object(phase, "run_tests", new_callable=AsyncMock) as mock_tests:
            mock_tests.return_value = {
                "passed": False,
                "failures": ["test_example.py::test_func"],
            }

            result = await phase.run()

            assert result["passed"] is False
            assert len(result.get("failures", [])) > 0
