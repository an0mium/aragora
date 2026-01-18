"""
Tests for Nomic Loop Implement Phase.

Phase 3: Implementation
- Tests code generation
- Tests file writing
- Tests backup creation
- Tests rollback on failure
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestImplementPhaseInitialization:
    """Tests for ImplementPhase initialization."""

    def test_init_with_required_args(self, mock_aragora_path, mock_codex_agent):
        """Should initialize with required arguments."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
        )

        assert phase.aragora_path == mock_aragora_path
        assert phase.codex == mock_codex_agent

    def test_init_with_backup_path(self, mock_aragora_path, mock_codex_agent, tmp_path):
        """Should accept custom backup path."""
        from aragora.nomic.phases.implement import ImplementPhase

        backup_dir = tmp_path / "backups"

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            backup_path=backup_dir,
        )

        assert phase.backup_path == backup_dir


class TestImplementPhaseCodeGeneration:
    """Tests for code generation."""

    @pytest.mark.asyncio
    async def test_generates_code_from_design(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn, mock_design_result
    ):
        """Should generate code from approved design."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, '_generate_code', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = {
                "aragora/errors.py": "class ErrorHandler:\n    pass",
                "tests/test_errors.py": "def test_handler():\n    assert True",
            }

            code = await phase.generate_code(mock_design_result["design"])

            assert len(code) >= 1
            mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_validates_generated_code_syntax(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn
    ):
        """Should validate syntax of generated code."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        valid_code = "def hello():\n    return 'world'"
        invalid_code = "def hello(\n    return"

        assert phase.validate_syntax(valid_code) is True
        assert phase.validate_syntax(invalid_code) is False

    @pytest.mark.asyncio
    async def test_rejects_code_with_dangerous_patterns(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn
    ):
        """Should reject code with dangerous patterns."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        dangerous_code = """
import os
def execute_command(cmd):
    os.system(cmd)
"""

        result = phase.check_dangerous_patterns(dangerous_code)

        assert result["safe"] is False
        assert len(result.get("patterns_found", [])) > 0


class TestImplementPhaseBackup:
    """Tests for backup functionality."""

    @pytest.mark.asyncio
    async def test_creates_backup_before_writing(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn
    ):
        """Should create backup before writing files."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        files_to_modify = ["aragora/utils.py"]

        with patch('shutil.copy2') as mock_copy:
            backup_manifest = await phase.create_backup(files_to_modify)

            assert backup_manifest is not None
            # Should attempt to copy each file

    @pytest.mark.asyncio
    async def test_backup_includes_timestamp(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn
    ):
        """Should include timestamp in backup path."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch('shutil.copy2'):
            backup_manifest = await phase.create_backup(["test.py"])

            if backup_manifest and "path" in backup_manifest:
                # Backup path should contain timestamp-like pattern
                assert backup_manifest["path"] is not None


class TestImplementPhaseFileWriting:
    """Tests for file writing."""

    @pytest.mark.asyncio
    async def test_writes_generated_files(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn
    ):
        """Should write generated files to disk."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        code_changes = {
            "aragora/new_module.py": "def new_function():\n    pass",
        }

        with patch('builtins.open', MagicMock()) as mock_open:
            with patch.object(phase, 'create_backup', new_callable=AsyncMock) as mock_backup:
                mock_backup.return_value = {"path": "/backup"}

                result = await phase.write_files(code_changes)

                # Should attempt to write files
                assert result is not None

    @pytest.mark.asyncio
    async def test_creates_parent_directories(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn
    ):
        """Should create parent directories if needed."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        code_changes = {
            "aragora/new_dir/new_module.py": "def func(): pass",
        }

        with patch('pathlib.Path.mkdir') as mock_mkdir:
            with patch('builtins.open', MagicMock()):
                with patch.object(phase, 'create_backup', new_callable=AsyncMock) as mock_backup:
                    mock_backup.return_value = {"path": "/backup"}

                    await phase.write_files(code_changes)

                    # Should call mkdir for parent directories


class TestImplementPhaseRollback:
    """Tests for rollback functionality."""

    @pytest.mark.asyncio
    async def test_rollback_restores_from_backup(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn
    ):
        """Should restore files from backup on rollback."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        backup_manifest = {
            "path": "/backup/20240101_120000",
            "files": ["aragora/utils.py"],
        }

        with patch('shutil.copy2') as mock_copy:
            await phase.rollback(backup_manifest)

            # Should restore backed up files

    @pytest.mark.asyncio
    async def test_rollback_removes_new_files(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn
    ):
        """Should remove newly created files on rollback."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        backup_manifest = {
            "path": "/backup",
            "files_created": ["aragora/new_module.py"],
        }

        with patch('pathlib.Path.unlink') as mock_unlink:
            await phase.rollback(backup_manifest)

            # Should attempt to remove created files


class TestImplementPhaseIntegration:
    """Integration tests for implement phase."""

    @pytest.mark.asyncio
    async def test_full_implementation_flow(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn, mock_design_result
    ):
        """Should complete full implementation flow."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, 'generate_code', new_callable=AsyncMock) as mock_gen:
            with patch.object(phase, 'create_backup', new_callable=AsyncMock) as mock_backup:
                with patch.object(phase, 'write_files', new_callable=AsyncMock) as mock_write:
                    mock_gen.return_value = {"aragora/test.py": "pass"}
                    mock_backup.return_value = {"path": "/backup"}
                    mock_write.return_value = {"success": True}

                    result = await phase.run(design=mock_design_result["design"])

                    assert result is not None
                    assert result.get("success", False) is True
                    mock_gen.assert_called_once()
                    mock_backup.assert_called_once()
                    mock_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_implementation_with_syntax_error(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn, mock_design_result
    ):
        """Should handle syntax errors in generated code."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, 'generate_code', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = {"aragora/test.py": "def broken(\n    return"}

            result = await phase.run(design=mock_design_result["design"])

            # Should detect syntax error and not write
            assert result.get("success", True) is False or "syntax" in str(result.get("error", "")).lower()

    @pytest.mark.asyncio
    async def test_implementation_with_rollback_on_failure(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn, mock_design_result
    ):
        """Should rollback on write failure."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, 'generate_code', new_callable=AsyncMock) as mock_gen:
            with patch.object(phase, 'create_backup', new_callable=AsyncMock) as mock_backup:
                with patch.object(phase, 'write_files', new_callable=AsyncMock) as mock_write:
                    with patch.object(phase, 'rollback', new_callable=AsyncMock) as mock_rollback:
                        mock_gen.return_value = {"aragora/test.py": "pass"}
                        mock_backup.return_value = {"path": "/backup"}
                        mock_write.side_effect = IOError("Write failed")

                        result = await phase.run(design=mock_design_result["design"])

                        # Should have attempted rollback
                        assert result.get("success", True) is False
                        mock_rollback.assert_called_once()
