"""
Comprehensive unit tests for ImplementPhase.

Tests the implementation phase module including:
- Initialization with various configurations
- Legacy API (run, generate_code, validate_syntax, create_backup, rollback)
- Modern execute() API with plan generation and execution
- Scope limiting and design gates
- Constitution compliance verification
- Git stash operations and crash recovery
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_aragora_path(tmp_path: Path) -> Path:
    """Create a mock aragora project structure."""
    (tmp_path / "aragora").mkdir()
    (tmp_path / "aragora" / "__init__.py").write_text('"""Aragora."""')
    (tmp_path / ".nomic").mkdir()
    (tmp_path / ".nomic" / "backups").mkdir()
    return tmp_path


@pytest.fixture
def mock_codex_agent():
    """Create a mock Codex agent."""
    agent = MagicMock()
    agent.name = "codex"
    agent.generate = AsyncMock(return_value="def new_function(): pass")
    return agent


@pytest.fixture
def mock_executor():
    """Create a mock task executor."""
    executor = MagicMock()

    @dataclass
    class TaskResult:
        success: bool
        error: str | None = None

    results = [TaskResult(success=True), TaskResult(success=True)]
    executor.execute_plan = AsyncMock(return_value=results)
    return executor


@pytest.fixture
def mock_plan():
    """Create a mock implementation plan."""
    plan = MagicMock()
    plan.design_hash = "abc123"
    plan.tasks = [
        MagicMock(id="task1", description="Create file"),
        MagicMock(id="task2", description="Modify file"),
    ]
    return plan


@pytest.fixture
def mock_plan_generator(mock_plan):
    """Create a mock plan generator."""

    async def generator(design, path):
        return mock_plan

    return AsyncMock(side_effect=generator)


@pytest.fixture
def mock_log_fn():
    """Create a mock log function."""
    return MagicMock()


@pytest.fixture
def mock_stream_emit_fn():
    """Create a mock stream emit function."""
    return MagicMock()


@pytest.fixture
def mock_design_gate():
    """Create a mock design gate."""
    gate = MagicMock()
    decision = MagicMock()
    decision.status = MagicMock()
    decision.status.value = "approved"
    decision.approver = "auto"
    gate.require_approval = AsyncMock(return_value=decision)
    return gate


@pytest.fixture
def mock_constitution_verifier():
    """Create a mock constitution verifier."""
    verifier = MagicMock()
    verifier.is_available = MagicMock(return_value=True)
    verifier.check_file_modification_allowed = MagicMock(return_value=(True, ""))
    return verifier


# ============================================================================
# ImplementPhase Initialization Tests
# ============================================================================


class TestImplementPhaseInitialization:
    """Tests for ImplementPhase initialization."""

    def test_init_with_required_args(self, mock_aragora_path):
        """Should initialize with required arguments."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(aragora_path=mock_aragora_path)

        assert phase.aragora_path == mock_aragora_path
        assert phase.protected_files is not None

    def test_init_with_codex_agent(self, mock_aragora_path, mock_codex_agent):
        """Should accept codex agent."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
        )

        assert phase.codex == mock_codex_agent

    def test_init_with_plan_generator(self, mock_aragora_path, mock_plan_generator, mock_executor):
        """Should accept plan generator and executor."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            plan_generator=mock_plan_generator,
            executor=mock_executor,
        )

        assert phase._plan_generator is mock_plan_generator
        assert phase._executor is mock_executor

    def test_init_with_protected_files(self, mock_aragora_path):
        """Should accept protected files list."""
        from aragora.nomic.phases.implement import ImplementPhase

        protected = ["secret.py", "config.py"]

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            protected_files=protected,
        )

        assert phase.protected_files == protected

    def test_init_with_default_protected_files(self, mock_aragora_path):
        """Should use default protected files."""
        from aragora.nomic.phases.implement import (
            ImplementPhase,
            DEFAULT_PROTECTED_FILES,
        )

        phase = ImplementPhase(aragora_path=mock_aragora_path)

        assert phase.protected_files == DEFAULT_PROTECTED_FILES
        assert "CLAUDE.md" in phase.protected_files

    def test_init_with_backup_path(self, mock_aragora_path, tmp_path):
        """Should accept custom backup path."""
        from aragora.nomic.phases.implement import ImplementPhase

        backup_dir = tmp_path / "custom_backup"

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            backup_path=backup_dir,
        )

        assert phase.backup_path == backup_dir

    def test_init_with_design_gate(self, mock_aragora_path, mock_design_gate):
        """Should accept design gate."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            design_gate=mock_design_gate,
        )

        assert phase._design_gate is mock_design_gate

    def test_init_with_constitution_verifier(self, mock_aragora_path, mock_constitution_verifier):
        """Should accept constitution verifier."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            constitution_verifier=mock_constitution_verifier,
        )

        assert phase._constitution_verifier is mock_constitution_verifier


# ============================================================================
# Legacy API Tests
# ============================================================================


class TestImplementPhaseGenerateCode:
    """Tests for generate_code method."""

    @pytest.mark.asyncio
    async def test_generates_with_codex(self, mock_aragora_path, mock_codex_agent, mock_log_fn):
        """Should use codex agent for code generation."""
        from aragora.nomic.phases.implement import ImplementPhase

        mock_codex_agent.generate = AsyncMock(return_value="def hello(): return 'world'")

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        code = await phase.generate_code("Create hello function")

        assert "generated.py" in code
        assert "def hello" in code["generated.py"]

    @pytest.mark.asyncio
    async def test_returns_empty_without_codex(self, mock_aragora_path, mock_log_fn):
        """Should return empty dict without codex agent."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        code = await phase.generate_code("Create something")

        assert code == {}


class TestImplementPhaseValidateSyntax:
    """Tests for validate_syntax method."""

    def test_validates_correct_syntax(self, mock_aragora_path):
        """Should pass valid Python syntax."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(aragora_path=mock_aragora_path)

        valid_code = """
def hello():
    return "world"

class MyClass:
    def method(self):
        pass
"""

        assert phase.validate_syntax(valid_code) is True

    def test_rejects_invalid_syntax(self, mock_aragora_path):
        """Should reject invalid Python syntax."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(aragora_path=mock_aragora_path)

        invalid_code = """
def hello(
    return "broken"
"""

        assert phase.validate_syntax(invalid_code) is False

    def test_handles_empty_code(self, mock_aragora_path):
        """Should handle empty code."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(aragora_path=mock_aragora_path)

        assert phase.validate_syntax("") is True
        assert phase.validate_syntax("   ") is True


class TestImplementPhaseCheckDangerousPatterns:
    """Tests for check_dangerous_patterns method."""

    def test_detects_os_system(self, mock_aragora_path):
        """Should detect os.system calls."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(aragora_path=mock_aragora_path)

        code = """
import os
os.system("rm -rf /")
"""

        result = phase.check_dangerous_patterns(code)

        assert result["safe"] is False
        assert any("os.system" in p["pattern"] for p in result["patterns_found"])

    def test_detects_eval(self, mock_aragora_path):
        """Should detect eval calls."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(aragora_path=mock_aragora_path)

        code = """
user_input = input()
eval(user_input)
"""

        result = phase.check_dangerous_patterns(code)

        assert result["safe"] is False
        assert any("eval(" in p["pattern"] for p in result["patterns_found"])

    def test_detects_exec(self, mock_aragora_path):
        """Should detect exec calls."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(aragora_path=mock_aragora_path)

        code = """
code = "print('hello')"
exec(code)
"""

        result = phase.check_dangerous_patterns(code)

        assert result["safe"] is False
        assert any("exec(" in p["pattern"] for p in result["patterns_found"])

    def test_detects_subprocess(self, mock_aragora_path):
        """Should detect subprocess calls."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(aragora_path=mock_aragora_path)

        code = """
import subprocess
subprocess.call(['ls', '-la'])
"""

        result = phase.check_dangerous_patterns(code)

        assert result["safe"] is False

    def test_allows_safe_code(self, mock_aragora_path):
        """Should allow safe code."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(aragora_path=mock_aragora_path)

        code = """
def process_data(data):
    return [x * 2 for x in data]
"""

        result = phase.check_dangerous_patterns(code)

        assert result["safe"] is True
        assert result["patterns_found"] == []


class TestImplementPhaseBackup:
    """Tests for backup functionality."""

    @pytest.mark.asyncio
    async def test_creates_backup_directory(self, mock_aragora_path, mock_log_fn):
        """Should create backup directory with timestamp."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        # Create a file to backup
        (mock_aragora_path / "aragora" / "test.py").write_text("# test")

        manifest = await phase.create_backup(["aragora/test.py"])

        assert manifest is not None
        assert "path" in manifest
        assert "timestamp" in manifest
        assert Path(manifest["path"]).exists()

    @pytest.mark.asyncio
    async def test_backup_copies_existing_files(self, mock_aragora_path, mock_log_fn):
        """Should copy existing files to backup."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        # Create files to backup
        (mock_aragora_path / "aragora" / "module.py").write_text("# module")

        manifest = await phase.create_backup(["aragora/module.py"])

        assert "aragora/module.py" in manifest["files"]
        backup_file = Path(manifest["path"]) / "aragora" / "module.py"
        assert backup_file.exists()

    @pytest.mark.asyncio
    async def test_backup_skips_nonexistent_files(self, mock_aragora_path, mock_log_fn):
        """Should skip files that don't exist."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        manifest = await phase.create_backup(["nonexistent.py"])

        assert "nonexistent.py" not in manifest.get("files", [])


class TestImplementPhaseWriteFiles:
    """Tests for write_files method."""

    @pytest.mark.asyncio
    async def test_writes_files(self, mock_aragora_path, mock_log_fn):
        """Should write code changes to files."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        code_changes = {
            "aragora/new_file.py": "def new_func(): pass",
        }

        result = await phase.write_files(code_changes)

        assert result["success"] is True
        assert "aragora/new_file.py" in result["files_written"]
        assert (mock_aragora_path / "aragora" / "new_file.py").exists()

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, mock_aragora_path, mock_log_fn):
        """Should create parent directories if needed."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        code_changes = {
            "aragora/new_dir/nested/file.py": "# new file",
        }

        result = await phase.write_files(code_changes)

        assert (mock_aragora_path / "aragora" / "new_dir" / "nested" / "file.py").exists()

    @pytest.mark.asyncio
    async def test_tracks_created_files(self, mock_aragora_path, mock_log_fn):
        """Should track newly created files."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )
        phase._backup_manifest = {"path": "/backup", "files": []}

        code_changes = {
            "aragora/brand_new.py": "# brand new",
        }

        result = await phase.write_files(code_changes)

        assert "aragora/brand_new.py" in result["files_created"]


class TestImplementPhaseRollback:
    """Tests for rollback functionality."""

    @pytest.mark.asyncio
    async def test_restores_backed_up_files(self, mock_aragora_path, mock_log_fn):
        """Should restore files from backup."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        # Create original file
        original_content = "# original content"
        (mock_aragora_path / "aragora" / "module.py").write_text(original_content)

        # Create backup
        manifest = await phase.create_backup(["aragora/module.py"])

        # Modify file
        (mock_aragora_path / "aragora" / "module.py").write_text("# modified")

        # Rollback
        await phase.rollback(manifest)

        # Check restoration
        restored = (mock_aragora_path / "aragora" / "module.py").read_text()
        assert restored == original_content

    @pytest.mark.asyncio
    async def test_removes_created_files(self, mock_aragora_path, mock_log_fn):
        """Should remove newly created files on rollback."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        # Create a new file
        new_file = mock_aragora_path / "aragora" / "new_created.py"
        new_file.write_text("# new file")

        manifest = {
            "path": str(mock_aragora_path / ".nomic" / "backups" / "test"),
            "files": [],
            "files_created": ["aragora/new_created.py"],
        }

        await phase.rollback(manifest)

        assert not new_file.exists()


class TestImplementPhaseRun:
    """Tests for legacy run() method."""

    @pytest.mark.asyncio
    async def test_complete_flow_success(self, mock_aragora_path, mock_codex_agent, mock_log_fn):
        """Should complete full implementation flow."""
        from aragora.nomic.phases.implement import ImplementPhase

        mock_codex_agent.generate = AsyncMock(return_value="def test(): pass")

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "generate_code", new_callable=AsyncMock) as mock_gen:
            with patch.object(phase, "create_backup", new_callable=AsyncMock) as mock_backup:
                with patch.object(phase, "write_files", new_callable=AsyncMock) as mock_write:
                    mock_gen.return_value = {"test.py": "def test(): pass"}
                    mock_backup.return_value = {"path": "/backup", "files": []}
                    mock_write.return_value = {"success": True, "files_written": ["test.py"]}

                    result = await phase.run(design="Create test function")

                    assert result["success"] is True
                    assert "test.py" in result["files_modified"]

    @pytest.mark.asyncio
    async def test_fails_on_syntax_error(self, mock_aragora_path, mock_codex_agent, mock_log_fn):
        """Should fail when generated code has syntax errors."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "generate_code", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = {"broken.py": "def broken(\n    return"}

            result = await phase.run(design="Create function")

            assert result["success"] is False
            assert "syntax" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_rollback_on_write_failure(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn
    ):
        """Should rollback on write failure."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "generate_code", new_callable=AsyncMock) as mock_gen:
            with patch.object(phase, "create_backup", new_callable=AsyncMock) as mock_backup:
                with patch.object(phase, "write_files", new_callable=AsyncMock) as mock_write:
                    with patch.object(phase, "rollback", new_callable=AsyncMock) as mock_rollback:
                        mock_gen.return_value = {"test.py": "pass"}
                        mock_backup.return_value = {"path": "/backup", "files": []}
                        mock_write.side_effect = OSError("Disk full")

                        result = await phase.run(design="Test")

                        assert result["success"] is False
                        mock_rollback.assert_called_once()


# ============================================================================
# Modern Execute API Tests
# ============================================================================


class TestImplementPhaseExecute:
    """Tests for execute() method."""

    @pytest.mark.asyncio
    async def test_execute_with_plan_generator(
        self,
        mock_aragora_path,
        mock_plan_generator,
        mock_executor,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should execute with plan generator."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            plan_generator=mock_plan_generator,
            executor=mock_executor,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch.object(phase, "_get_git_diff", new_callable=AsyncMock) as mock_diff:
            with patch.object(phase, "_get_modified_files", new_callable=AsyncMock) as mock_files:
                with patch.object(phase, "_git_stash_create", new_callable=AsyncMock) as mock_stash:
                    with patch.object(
                        phase, "_verify_constitution_compliance", new_callable=AsyncMock
                    ) as mock_verify:
                        mock_diff.return_value = "file.py | 10 ++"
                        mock_files.return_value = ["aragora/file.py"]
                        mock_stash.return_value = "stash@{0}"
                        mock_verify.return_value = None

                        result = await phase.execute(design="Create new module")

                        assert result["success"] is True
                        mock_plan_generator.assert_called_once()
                        mock_executor.execute_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_falls_back_to_legacy(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn
    ):
        """Should fall back to legacy mode without plan generator."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_legacy_implement", new_callable=AsyncMock) as mock_legacy:
            mock_legacy.return_value = MagicMock(
                success=True,
                data={},
                duration_seconds=10,
                files_modified=[],
                diff_summary="",
            )

            with patch.dict("os.environ", {"ARAGORA_HYBRID_IMPLEMENT": "1"}):
                await phase.execute(design="Simple task")

            mock_legacy.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_emits_events(self, mock_aragora_path, mock_log_fn, mock_stream_emit_fn):
        """Should emit phase start/end events."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch.object(phase, "_legacy_implement", new_callable=AsyncMock) as mock_legacy:
            mock_legacy.return_value = MagicMock(
                success=True,
                data={},
                duration_seconds=10,
                files_modified=[],
                diff_summary="",
            )

            with patch.dict("os.environ", {"ARAGORA_HYBRID_IMPLEMENT": "1"}):
                await phase.execute(design="Test")

        start_calls = [c for c in mock_stream_emit_fn.call_args_list if c[0][0] == "on_phase_start"]
        assert len(start_calls) >= 1


# ============================================================================
# Scope Limiting Tests
# ============================================================================


class TestImplementPhaseScopeLimiting:
    """Tests for scope limiting functionality."""

    @pytest.mark.asyncio
    async def test_rejects_overly_complex_design(
        self, mock_aragora_path, mock_log_fn, mock_stream_emit_fn
    ):
        """Should reject designs that exceed scope."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        # Very complex design
        complex_design = """
Create 50 new files across all modules.
Refactor the entire codebase.
Add new APIs, database migrations, and frontend changes.
""" + "\n".join([f"- aragora/module_{i}.py" for i in range(50)])

        with patch.dict(
            "os.environ", {"ARAGORA_SCOPE_CHECK": "1", "ARAGORA_HYBRID_IMPLEMENT": "1"}
        ):
            with patch("aragora.nomic.phases.implement.ScopeLimiter") as mock_limiter_class:
                mock_limiter = MagicMock()
                mock_eval = MagicMock()
                mock_eval.is_implementable = False
                mock_eval.reason = "Too many files"
                mock_eval.risk_factors = ["50 files affected"]
                mock_eval.suggested_simplifications = ["Break into smaller tasks"]
                mock_eval.to_dict = MagicMock(return_value={"is_implementable": False})
                mock_limiter.evaluate.return_value = mock_eval
                mock_limiter_class.return_value = mock_limiter

                result = await phase.execute(design=complex_design)

                assert result["success"] is False
                assert (
                    "scope" in result.get("error", "").lower()
                    or "complex" in result.get("error", "").lower()
                )


# ============================================================================
# Design Gate Tests
# ============================================================================


class TestImplementPhaseDesignGate:
    """Tests for design gate functionality."""

    @pytest.mark.asyncio
    async def test_approved_by_gate(
        self,
        mock_aragora_path,
        mock_plan_generator,
        mock_executor,
        mock_design_gate,
        mock_log_fn,
    ):
        """Should proceed when design gate approves."""
        from aragora.nomic.phases.implement import ImplementPhase
        from aragora.nomic.gates import ApprovalStatus

        mock_decision = MagicMock()
        mock_decision.status = ApprovalStatus.APPROVED
        mock_decision.approver = "auto"
        mock_design_gate.require_approval = AsyncMock(return_value=mock_decision)

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            plan_generator=mock_plan_generator,
            executor=mock_executor,
            design_gate=mock_design_gate,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_get_git_diff", new_callable=AsyncMock) as mock_diff:
            with patch.object(phase, "_get_modified_files", new_callable=AsyncMock) as mock_files:
                with patch.object(phase, "_git_stash_create", new_callable=AsyncMock):
                    with patch.object(
                        phase, "_verify_constitution_compliance", new_callable=AsyncMock
                    ) as mock_verify:
                        mock_diff.return_value = ""
                        mock_files.return_value = []
                        mock_verify.return_value = None

                        with patch.dict(
                            "os.environ",
                            {"ARAGORA_SCOPE_CHECK": "1", "ARAGORA_HYBRID_IMPLEMENT": "1"},
                        ):
                            with patch(
                                "aragora.nomic.phases.implement.ScopeLimiter"
                            ) as mock_limiter_class:
                                mock_limiter = MagicMock()
                                mock_eval = MagicMock()
                                mock_eval.is_implementable = True
                                mock_eval.complexity_score = 0.5
                                mock_eval.file_count = 2
                                mock_eval.risk_factors = []
                                mock_limiter.evaluate.return_value = mock_eval
                                mock_limiter_class.return_value = mock_limiter

                                result = await phase.execute(design="Test")

                        mock_design_gate.require_approval.assert_called_once()
                        # Design gate approved, so execution should proceed


# ============================================================================
# Constitution Compliance Tests
# ============================================================================


class TestImplementPhaseConstitutionCompliance:
    """Tests for constitution compliance verification."""

    @pytest.mark.asyncio
    async def test_blocks_protected_file_modifications(self, mock_aragora_path, mock_log_fn):
        """Should block modifications to protected files."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            protected_files=["CLAUDE.md", "aragora/__init__.py"],
            log_fn=mock_log_fn,
        )

        violation = await phase._verify_constitution_compliance(
            modified_files=["CLAUDE.md", "aragora/utils.py"],
            diff="",
        )

        assert violation is not None
        assert "protected" in violation.lower()

    @pytest.mark.asyncio
    async def test_allows_safe_modifications(self, mock_aragora_path, mock_log_fn):
        """Should allow modifications to non-protected files."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            protected_files=["CLAUDE.md"],
            log_fn=mock_log_fn,
        )

        violation = await phase._verify_constitution_compliance(
            modified_files=["aragora/utils.py", "aragora/new_module.py"],
            diff="",
        )

        assert violation is None

    @pytest.mark.asyncio
    async def test_uses_constitution_verifier(
        self, mock_aragora_path, mock_constitution_verifier, mock_log_fn
    ):
        """Should use constitution verifier when available."""
        from aragora.nomic.phases.implement import ImplementPhase

        mock_constitution_verifier.check_file_modification_allowed.return_value = (
            False,
            "Violates rule X",
        )

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            constitution_verifier=mock_constitution_verifier,
            protected_files=[],  # No simple protected files
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"diff output", b""))
            mock_exec.return_value = mock_proc

            violation = await phase._verify_constitution_compliance(
                modified_files=["aragora/some_file.py"],
                diff="some diff",
            )

        assert violation is not None
        # The message contains the rule text (case may vary)
        assert "rule x" in violation.lower() or "violates" in violation.lower()


# ============================================================================
# Git Stash Tests
# ============================================================================


class TestImplementPhaseGitStash:
    """Tests for git stash operations."""

    @pytest.mark.asyncio
    async def test_creates_stash(self, mock_aragora_path, mock_log_fn):
        """Should create git stash."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"stash@{0}", b""))
            mock_exec.return_value = mock_proc

            stash_ref = await phase._git_stash_create()

            assert stash_ref == "stash@{0}"

    @pytest.mark.asyncio
    async def test_handles_stash_failure(self, mock_aragora_path, mock_log_fn):
        """Should handle stash creation failure."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = OSError("git not found")

            stash_ref = await phase._git_stash_create()

            assert stash_ref is None

    @pytest.mark.asyncio
    async def test_applies_stash(self, mock_aragora_path, mock_log_fn):
        """Should apply git stash."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc

            result = await phase._git_stash_pop("stash@{0}")

            assert result is True

    @pytest.mark.asyncio
    async def test_handles_no_stash_ref(self, mock_aragora_path, mock_log_fn):
        """Should handle None stash ref."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        result = await phase._git_stash_pop(None)

        assert result is False


# ============================================================================
# Git Diff and Modified Files Tests
# ============================================================================


class TestImplementPhaseGitOperations:
    """Tests for git operations."""

    @pytest.mark.asyncio
    async def test_gets_git_diff(self, mock_aragora_path, mock_log_fn):
        """Should get git diff."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(
                return_value=(b"file.py | 10 ++\nother.py | 5 --", b"")
            )
            mock_exec.return_value = mock_proc

            diff = await phase._get_git_diff()

            assert "file.py" in diff
            assert "other.py" in diff

    @pytest.mark.asyncio
    async def test_gets_modified_files(self, mock_aragora_path, mock_log_fn):
        """Should get list of modified files."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(
                return_value=(b"aragora/file1.py\naragora/file2.py", b"")
            )
            mock_exec.return_value = mock_proc

            files = await phase._get_modified_files()

            assert "aragora/file1.py" in files
            assert "aragora/file2.py" in files

    @pytest.mark.asyncio
    async def test_handles_git_error(self, mock_aragora_path, mock_log_fn):
        """Should handle git command errors."""
        from aragora.nomic.phases.implement import ImplementPhase

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = OSError("git not found")

            diff = await phase._get_git_diff()
            files = await phase._get_modified_files()

            assert diff == ""
            assert files == []


# ============================================================================
# Crash Recovery Tests
# ============================================================================


class TestImplementPhaseCrashRecovery:
    """Tests for crash recovery functionality."""

    @pytest.mark.asyncio
    async def test_resumes_from_checkpoint(
        self,
        mock_aragora_path,
        mock_plan,
        mock_executor,
        mock_log_fn,
    ):
        """Should resume from checkpoint if available."""
        from aragora.nomic.phases.implement import ImplementPhase

        progress = MagicMock()
        progress.plan = mock_plan
        progress.completed_tasks = ["task1"]
        progress.git_stash_ref = "stash@{0}"

        progress_loader = MagicMock(return_value=progress)

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            executor=mock_executor,
            progress_loader=progress_loader,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_get_git_diff", new_callable=AsyncMock) as mock_diff:
            with patch.object(phase, "_get_modified_files", new_callable=AsyncMock) as mock_files:
                with patch.object(
                    phase, "_verify_constitution_compliance", new_callable=AsyncMock
                ) as mock_verify:
                    mock_diff.return_value = ""
                    mock_files.return_value = []
                    mock_verify.return_value = None

                    with patch.dict(
                        "os.environ",
                        {"ARAGORA_SCOPE_CHECK": "0", "ARAGORA_HYBRID_IMPLEMENT": "1"},
                    ):
                        # Use the same design hash
                        import hashlib

                        design = "Test design"
                        mock_plan.design_hash = hashlib.md5(
                            design.encode(), usedforsecurity=False
                        ).hexdigest()

                        await phase.execute(design=design)

                        # Should have used the progress loader
                        progress_loader.assert_called()


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestImplementPhaseErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_plan_generation_failure(
        self, mock_aragora_path, mock_log_fn, mock_stream_emit_fn
    ):
        """Should handle plan generation failure."""
        from aragora.nomic.phases.implement import ImplementPhase

        failing_generator = AsyncMock(side_effect=Exception("Plan failed"))

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            plan_generator=failing_generator,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch.dict(
            "os.environ", {"ARAGORA_SCOPE_CHECK": "0", "ARAGORA_HYBRID_IMPLEMENT": "1"}
        ):
            result = await phase.execute(design="Test")

            assert result["success"] is False
            assert "Plan failed" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_handles_executor_failure(
        self,
        mock_aragora_path,
        mock_plan_generator,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should handle executor failure and rollback."""
        from aragora.nomic.phases.implement import ImplementPhase

        failing_executor = MagicMock()
        failing_executor.execute_plan = AsyncMock(side_effect=Exception("Execution crashed"))

        phase = ImplementPhase(
            aragora_path=mock_aragora_path,
            plan_generator=mock_plan_generator,
            executor=failing_executor,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch.object(phase, "_git_stash_create", new_callable=AsyncMock) as mock_stash:
            with patch.object(phase, "_git_stash_pop", new_callable=AsyncMock) as mock_pop:
                mock_stash.return_value = "stash@{0}"

                with patch.dict(
                    "os.environ",
                    {"ARAGORA_SCOPE_CHECK": "0", "ARAGORA_HYBRID_IMPLEMENT": "1"},
                ):
                    result = await phase.execute(design="Test")

                    assert result["success"] is False
                    # Should have attempted rollback
                    mock_pop.assert_called_with("stash@{0}")
