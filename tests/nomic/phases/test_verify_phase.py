"""
Comprehensive unit tests for VerifyPhase.

Tests the verification phase module including:
- Initialization with various configurations
- Syntax checking via py_compile
- Import validation
- Test execution with pytest
- Codex audit functionality
- Evidence staleness checking
- Test quality gate integration
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_aragora_path(tmp_path: Path) -> Path:
    """Create a mock aragora project structure."""
    (tmp_path / "aragora").mkdir()
    (tmp_path / "aragora" / "__init__.py").write_text('"""Aragora."""')
    (tmp_path / "tests").mkdir()
    return tmp_path


@pytest.fixture
def mock_codex_agent():
    """Create a mock Codex agent."""
    agent = MagicMock()
    agent.name = "codex"
    agent.generate = AsyncMock(
        return_value="""
CODE QUALITY: 8/10
TEST COVERAGE: 7/10
DESIGN ALIGNMENT: 9/10
RISK ASSESSMENT: 2/10
VERDICT: APPROVE - Changes look good
"""
    )
    return agent


@pytest.fixture
def mock_nomic_integration():
    """Create a mock nomic integration."""
    integration = MagicMock()
    integration.checkpoint = AsyncMock()
    return integration


@pytest.fixture
def mock_log_fn():
    """Create a mock log function."""
    return MagicMock()


@pytest.fixture
def mock_stream_emit_fn():
    """Create a mock stream emit function."""
    return MagicMock()


@pytest.fixture
def mock_save_state_fn():
    """Create a mock save state function."""
    return MagicMock()


@pytest.fixture
def mock_test_quality_gate():
    """Create a mock test quality gate."""
    gate = MagicMock()
    decision = MagicMock()
    decision.status = MagicMock()
    decision.status.value = "approved"
    decision.reason = "Tests pass quality threshold"
    decision.to_dict = MagicMock(return_value={"status": "approved"})
    gate.require_approval = AsyncMock(return_value=decision)
    return gate


# ============================================================================
# VerifyPhase Initialization Tests
# ============================================================================


class TestVerifyPhaseInitialization:
    """Tests for VerifyPhase initialization."""

    def test_init_with_required_args(self, mock_aragora_path):
        """Should initialize with required arguments."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(aragora_path=mock_aragora_path)

        assert phase.aragora_path == mock_aragora_path

    def test_init_with_codex_agent(self, mock_aragora_path, mock_codex_agent):
        """Should accept codex agent for audit."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            codex=mock_codex_agent,
        )

        assert phase.codex == mock_codex_agent

    def test_init_with_nomic_integration(self, mock_aragora_path, mock_nomic_integration):
        """Should accept nomic integration."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            nomic_integration=mock_nomic_integration,
        )

        assert phase.nomic_integration == mock_nomic_integration

    def test_init_with_cycle_count(self, mock_aragora_path):
        """Should accept cycle count."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            cycle_count=5,
        )

        assert phase.cycle_count == 5

    def test_init_with_callbacks(
        self, mock_aragora_path, mock_log_fn, mock_stream_emit_fn, mock_save_state_fn
    ):
        """Should accept callback functions."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
            save_state_fn=mock_save_state_fn,
        )

        assert phase._log is mock_log_fn
        assert phase._stream_emit is mock_stream_emit_fn
        assert phase._save_state is mock_save_state_fn

    def test_init_with_test_quality_gate(self, mock_aragora_path, mock_test_quality_gate):
        """Should accept test quality gate."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            test_quality_gate=mock_test_quality_gate,
        )

        assert phase._test_quality_gate == mock_test_quality_gate


# ============================================================================
# Syntax Check Tests
# ============================================================================


class TestVerifyPhaseSyntaxCheck:
    """Tests for _check_syntax method."""

    @pytest.mark.asyncio
    async def test_syntax_check_passes(self, mock_aragora_path, mock_log_fn):
        """Should pass when syntax is valid."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc

            result = await phase._check_syntax()

            assert result["check"] == "syntax"
            assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_syntax_check_fails(self, mock_aragora_path, mock_log_fn):
        """Should fail when syntax is invalid."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(
                return_value=(b"", b"SyntaxError: invalid syntax at line 5")
            )
            mock_exec.return_value = mock_proc

            result = await phase._check_syntax()

            assert result["check"] == "syntax"
            assert result["passed"] is False
            assert "SyntaxError" in result.get("output", "")

    @pytest.mark.asyncio
    async def test_syntax_check_exception(self, mock_aragora_path, mock_log_fn):
        """Should handle exceptions gracefully."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = OSError("python not found")

            result = await phase._check_syntax()

            assert result["check"] == "syntax"
            assert result["passed"] is False
            assert "error" in result


# ============================================================================
# Import Check Tests
# ============================================================================


class TestVerifyPhaseImportCheck:
    """Tests for _check_imports method."""

    @pytest.mark.asyncio
    async def test_import_check_passes(self, mock_aragora_path, mock_log_fn):
        """Should pass when import succeeds."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"OK", b""))
            mock_exec.return_value = mock_proc

            result = await phase._check_imports()

            assert result["check"] == "import"
            assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_import_check_fails(self, mock_aragora_path, mock_log_fn):
        """Should fail when import fails."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(
                return_value=(b"", b"ModuleNotFoundError: No module named 'missing'")
            )
            mock_exec.return_value = mock_proc

            result = await phase._check_imports()

            assert result["check"] == "import"
            assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_import_check_timeout(self, mock_aragora_path, mock_log_fn):
        """Should handle import timeout."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_exec.return_value = mock_proc

            result = await phase._check_imports()

            assert result["check"] == "import"
            assert result["passed"] is False
            assert result.get("error") == "timeout"


# ============================================================================
# Test Execution Tests
# ============================================================================


class TestVerifyPhaseRunTests:
    """Tests for _run_tests method."""

    @pytest.mark.asyncio
    async def test_tests_pass(self, mock_aragora_path, mock_log_fn):
        """Should pass when all tests pass."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(
                return_value=(b"====== 10 passed in 2.5s ======", b"")
            )
            mock_exec.return_value = mock_proc

            result = await phase._run_tests()

            assert result["check"] == "tests"
            assert result["passed"] is True
            assert "10 passed" in result.get("output", "")

    @pytest.mark.asyncio
    async def test_tests_fail(self, mock_aragora_path, mock_log_fn):
        """Should fail when tests fail."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(
                return_value=(
                    b"FAILED test_module.py::test_func - AssertionError",
                    b"",
                )
            )
            mock_exec.return_value = mock_proc

            result = await phase._run_tests()

            assert result["check"] == "tests"
            assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_no_tests_collected(self, mock_aragora_path, mock_log_fn):
        """Should pass when no tests are collected (exit code 5)."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 5  # pytest "no tests ran" exit code
            mock_proc.communicate = AsyncMock(return_value=(b"no tests ran in 0.1s", b""))
            mock_exec.return_value = mock_proc

            result = await phase._run_tests()

            assert result["check"] == "tests"
            assert result["passed"] is True
            assert "no tests" in result.get("note", "").lower()

    @pytest.mark.asyncio
    async def test_tests_timeout(self, mock_aragora_path, mock_log_fn):
        """Should handle test execution timeout."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_exec.return_value = mock_proc

            result = await phase._run_tests()

            assert result["check"] == "tests"
            assert result["passed"] is False
            assert result.get("error") == "timeout"


# ============================================================================
# Codex Audit Tests
# ============================================================================


class TestVerifyPhaseCodexAudit:
    """Tests for _codex_audit method."""

    @pytest.mark.asyncio
    async def test_audit_approves(self, mock_aragora_path, mock_codex_agent, mock_log_fn):
        """Should pass when codex audit approves."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            codex=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_get_changed_files", new_callable=AsyncMock) as mock_files:
            mock_files.return_value = ["aragora/module.py"]

            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"diff content here", b""))
                mock_exec.return_value = mock_proc

                with patch("aragora.server.stream.arena_hooks.streaming_task_context") as mock_ctx:
                    mock_ctx.return_value.__enter__ = MagicMock()
                    mock_ctx.return_value.__exit__ = MagicMock()

                    result = await phase._codex_audit()

                    assert result is not None
                    assert result["check"] == "codex_audit"
                    assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_audit_raises_concerns(self, mock_aragora_path, mock_codex_agent, mock_log_fn):
        """Should fail when codex audit has concerns."""
        from aragora.nomic.phases.verify import VerifyPhase

        mock_codex_agent.generate = AsyncMock(
            return_value="""
CODE QUALITY: 3/10
TEST COVERAGE: 2/10
DESIGN ALIGNMENT: 4/10
RISK ASSESSMENT: 8/10
VERDICT: CONCERNS - High risk of runtime errors, missing error handling
"""
        )

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            codex=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_get_changed_files", new_callable=AsyncMock) as mock_files:
            mock_files.return_value = ["aragora/module.py"]

            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"diff", b""))
                mock_exec.return_value = mock_proc

                with patch("aragora.server.stream.arena_hooks.streaming_task_context") as mock_ctx:
                    mock_ctx.return_value.__enter__ = MagicMock()
                    mock_ctx.return_value.__exit__ = MagicMock()

                    result = await phase._codex_audit()

                    assert result is not None
                    assert result["check"] == "codex_audit"
                    assert result["passed"] is False
                    assert "concerns" in result.get("note", "").lower()

    @pytest.mark.asyncio
    async def test_audit_handles_error(self, mock_aragora_path, mock_codex_agent, mock_log_fn):
        """Should handle audit errors gracefully."""
        from aragora.nomic.phases.verify import VerifyPhase

        mock_codex_agent.generate = AsyncMock(side_effect=ConnectionError("API down"))

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            codex=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_get_changed_files", new_callable=AsyncMock) as mock_files:
            mock_files.return_value = ["test.py"]

            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_exec.return_value = mock_proc

                with patch("aragora.server.stream.arena_hooks.streaming_task_context") as mock_ctx:
                    mock_ctx.return_value.__enter__ = MagicMock()
                    mock_ctx.return_value.__exit__ = MagicMock()

                    result = await phase._codex_audit()

                    # Should pass despite error (don't block on audit failure)
                    assert result is not None
                    assert result["passed"] is True
                    assert "error" in result


# ============================================================================
# Changed Files Tests
# ============================================================================


class TestVerifyPhaseChangedFiles:
    """Tests for _get_changed_files method."""

    @pytest.mark.asyncio
    async def test_gets_changed_files(self, mock_aragora_path, mock_log_fn):
        """Should get list of changed files from git."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(
                return_value=(b"aragora/file1.py\naragora/file2.py\ntests/test.py", b"")
            )
            mock_exec.return_value = mock_proc

            files = await phase._get_changed_files()

            assert "aragora/file1.py" in files
            assert "aragora/file2.py" in files
            assert "tests/test.py" in files

    @pytest.mark.asyncio
    async def test_handles_no_changes(self, mock_aragora_path, mock_log_fn):
        """Should handle no changes."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc

            files = await phase._get_changed_files()

            assert files == []

    @pytest.mark.asyncio
    async def test_handles_git_error(self, mock_aragora_path, mock_log_fn):
        """Should handle git command errors."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = OSError("git not found")

            files = await phase._get_changed_files()

            assert files == []


# ============================================================================
# Staleness Check Tests
# ============================================================================


class TestVerifyPhaseStalenessCheck:
    """Tests for _check_staleness method."""

    @pytest.mark.asyncio
    async def test_checkpoints_with_integration(
        self, mock_aragora_path, mock_nomic_integration, mock_log_fn
    ):
        """Should checkpoint verify phase via nomic integration."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            nomic_integration=mock_nomic_integration,
            cycle_count=3,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_get_changed_files", new_callable=AsyncMock) as mock_files:
            mock_files.return_value = ["aragora/module.py"]

            result = await phase._check_staleness()

            mock_nomic_integration.checkpoint.assert_called_once()
            call_kwargs = mock_nomic_integration.checkpoint.call_args.kwargs
            assert call_kwargs["phase"] == "verify"
            assert call_kwargs["cycle"] == 3

    @pytest.mark.asyncio
    async def test_handles_checkpoint_error(
        self, mock_aragora_path, mock_nomic_integration, mock_log_fn
    ):
        """Should handle checkpoint errors gracefully."""
        from aragora.nomic.phases.verify import VerifyPhase

        mock_nomic_integration.checkpoint = AsyncMock(side_effect=RuntimeError("Checkpoint failed"))

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            nomic_integration=mock_nomic_integration,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_get_changed_files", new_callable=AsyncMock) as mock_files:
            mock_files.return_value = []

            # Should not raise
            result = await phase._check_staleness()
            assert result == []


# ============================================================================
# Execute Tests
# ============================================================================


class TestVerifyPhaseExecute:
    """Tests for execute() method."""

    @pytest.mark.asyncio
    async def test_execute_all_pass(
        self, mock_aragora_path, mock_log_fn, mock_stream_emit_fn, mock_save_state_fn
    ):
        """Should pass when all checks pass."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
            save_state_fn=mock_save_state_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {
                        "check": "tests",
                        "passed": True,
                        "output": "10 passed",
                    }

                    result = await phase.execute()

                    assert result["success"] is True
                    assert result["tests_passed"] is True
                    assert result["syntax_valid"] is True

    @pytest.mark.asyncio
    async def test_execute_syntax_fails(self, mock_aragora_path, mock_log_fn, mock_save_state_fn):
        """Should fail when syntax check fails."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            save_state_fn=mock_save_state_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {
                        "check": "syntax",
                        "passed": False,
                        "output": "SyntaxError",
                    }
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {"check": "tests", "passed": True}

                    result = await phase.execute()

                    assert result["success"] is False
                    assert result["syntax_valid"] is False

    @pytest.mark.asyncio
    async def test_execute_tests_fail(self, mock_aragora_path, mock_log_fn, mock_save_state_fn):
        """Should fail when tests fail."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            save_state_fn=mock_save_state_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {
                        "check": "tests",
                        "passed": False,
                        "output": "FAILED test_func",
                    }

                    result = await phase.execute()

                    assert result["success"] is False

    @pytest.mark.asyncio
    async def test_execute_with_codex_audit(
        self, mock_aragora_path, mock_codex_agent, mock_log_fn, mock_save_state_fn
    ):
        """Should run codex audit when codex is available."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            codex=mock_codex_agent,
            log_fn=mock_log_fn,
            save_state_fn=mock_save_state_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    with patch.object(phase, "_codex_audit", new_callable=AsyncMock) as mock_audit:
                        mock_syntax.return_value = {"check": "syntax", "passed": True}
                        mock_imports.return_value = {"check": "import", "passed": True}
                        mock_tests.return_value = {"check": "tests", "passed": True}
                        mock_audit.return_value = {
                            "check": "codex_audit",
                            "passed": True,
                        }

                        result = await phase.execute()

                        mock_audit.assert_called_once()
                        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_emits_events(
        self, mock_aragora_path, mock_log_fn, mock_stream_emit_fn, mock_save_state_fn
    ):
        """Should emit phase start/end and verification events."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
            save_state_fn=mock_save_state_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {"check": "tests", "passed": True}

                    await phase.execute()

                    # Check phase events
                    start_calls = [
                        c for c in mock_stream_emit_fn.call_args_list if c[0][0] == "on_phase_start"
                    ]
                    assert len(start_calls) >= 1
                    assert start_calls[0][0][1] == "verify"

                    # Check verification start event
                    verify_start_calls = [
                        c
                        for c in mock_stream_emit_fn.call_args_list
                        if c[0][0] == "on_verification_start"
                    ]
                    assert len(verify_start_calls) >= 1

    @pytest.mark.asyncio
    async def test_execute_saves_state(self, mock_aragora_path, mock_log_fn, mock_save_state_fn):
        """Should save state during execution."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            save_state_fn=mock_save_state_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {"check": "tests", "passed": True}

                    await phase.execute()

                    mock_save_state_fn.assert_called()
                    call_args = mock_save_state_fn.call_args[0][0]
                    assert call_args["phase"] == "verify"
                    assert call_args["stage"] == "complete"


# ============================================================================
# Test Quality Gate Tests
# ============================================================================


class TestVerifyPhaseTestQualityGate:
    """Tests for test quality gate integration."""

    @pytest.mark.asyncio
    async def test_gate_approves(
        self,
        mock_aragora_path,
        mock_test_quality_gate,
        mock_log_fn,
        mock_save_state_fn,
    ):
        """Should proceed when quality gate approves."""
        from aragora.nomic.phases.verify import VerifyPhase
        from aragora.nomic.gates import ApprovalStatus

        mock_decision = MagicMock()
        mock_decision.status = ApprovalStatus.APPROVED
        mock_decision.reason = "Quality checks pass"
        mock_decision.to_dict = MagicMock(return_value={"status": "approved"})
        mock_test_quality_gate.require_approval = AsyncMock(return_value=mock_decision)

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            test_quality_gate=mock_test_quality_gate,
            log_fn=mock_log_fn,
            save_state_fn=mock_save_state_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {
                        "check": "tests",
                        "passed": True,
                        "output": "passed",
                    }

                    result = await phase.execute()

                    mock_test_quality_gate.require_approval.assert_called_once()
                    assert result["success"] is True
                    assert "quality_gate" in result["data"]

    @pytest.mark.asyncio
    async def test_gate_rejects(
        self,
        mock_aragora_path,
        mock_test_quality_gate,
        mock_log_fn,
        mock_save_state_fn,
    ):
        """Should fail when quality gate rejects."""
        from aragora.nomic.phases.verify import VerifyPhase
        from aragora.nomic.gates import ApprovalRequired

        mock_test_quality_gate.require_approval = AsyncMock(
            side_effect=ApprovalRequired("Coverage too low", recoverable=True)
        )

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            test_quality_gate=mock_test_quality_gate,
            log_fn=mock_log_fn,
            save_state_fn=mock_save_state_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {"check": "tests", "passed": True}

                    result = await phase.execute()

                    assert result["success"] is False
                    assert result["tests_passed"] is False

    @pytest.mark.asyncio
    async def test_gate_skipped_when_disabled(
        self,
        mock_aragora_path,
        mock_test_quality_gate,
        mock_log_fn,
        mock_save_state_fn,
    ):
        """Should skip gate when status is SKIPPED."""
        from aragora.nomic.phases.verify import VerifyPhase
        from aragora.nomic.gates import ApprovalStatus

        mock_decision = MagicMock()
        mock_decision.status = ApprovalStatus.SKIPPED
        mock_decision.to_dict = MagicMock(return_value={"status": "skipped"})
        mock_test_quality_gate.require_approval = AsyncMock(return_value=mock_decision)

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            test_quality_gate=mock_test_quality_gate,
            log_fn=mock_log_fn,
            save_state_fn=mock_save_state_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {"check": "tests", "passed": True}

                    result = await phase.execute()

                    # Should still succeed even with skipped gate
                    assert result["success"] is True


# ============================================================================
# Result Structure Tests
# ============================================================================


class TestVerifyPhaseResultStructure:
    """Tests for VerifyResult structure."""

    @pytest.mark.asyncio
    async def test_result_has_required_fields(
        self, mock_aragora_path, mock_log_fn, mock_save_state_fn
    ):
        """Should return result with all required fields."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            save_state_fn=mock_save_state_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {
                        "check": "tests",
                        "passed": True,
                        "output": "passed",
                    }

                    result = await phase.execute()

                    # Check required fields from VerifyResult
                    assert "success" in result
                    assert "data" in result
                    assert "duration_seconds" in result
                    assert "tests_passed" in result
                    assert "test_output" in result
                    assert "syntax_valid" in result

    @pytest.mark.asyncio
    async def test_result_contains_checks(self, mock_aragora_path, mock_log_fn, mock_save_state_fn):
        """Should include all check results in data."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            save_state_fn=mock_save_state_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {
                        "check": "syntax",
                        "passed": True,
                        "output": "",
                    }
                    mock_imports.return_value = {
                        "check": "import",
                        "passed": True,
                        "output": "",
                    }
                    mock_tests.return_value = {
                        "check": "tests",
                        "passed": True,
                        "output": "10 passed",
                    }

                    result = await phase.execute()

                    checks = result["data"]["checks"]
                    assert len(checks) == 3
                    check_names = [c["check"] for c in checks]
                    assert "syntax" in check_names
                    assert "import" in check_names
                    assert "tests" in check_names

    @pytest.mark.asyncio
    async def test_result_duration_tracked(
        self, mock_aragora_path, mock_log_fn, mock_save_state_fn
    ):
        """Should track execution duration."""
        from aragora.nomic.phases.verify import VerifyPhase

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=mock_log_fn,
            save_state_fn=mock_save_state_fn,
        )

        with patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax:
            with patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports:
                with patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests:
                    mock_syntax.return_value = {"check": "syntax", "passed": True}
                    mock_imports.return_value = {"check": "import", "passed": True}
                    mock_tests.return_value = {"check": "tests", "passed": True}

                    result = await phase.execute()

                    assert result["duration_seconds"] >= 0
                    assert isinstance(result["duration_seconds"], float)
