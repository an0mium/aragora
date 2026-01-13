"""
Tests for Nomic Loop phases and pre-flight checks.

Tests the modular phase system and pre-flight health check infrastructure.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from scripts.nomic.preflight import (
    PreflightCheck,
    PreflightReport,
    check_api_keys,
    check_disk_space,
    check_database_connectivity,
    check_protected_files,
    check_git_repository,
    check_backup_directory,
    run_preflight_checks,
)


class TestPreflightCheck:
    """Tests for PreflightCheck class."""

    def test_passing_check(self):
        check = PreflightCheck(name="Test Check", passed=True, message="All good", critical=True)
        assert check.passed
        assert check.critical
        assert "PASS" in repr(check)

    def test_failing_critical_check(self):
        check = PreflightCheck(
            name="Critical Check", passed=False, message="Something failed", critical=True
        )
        assert not check.passed
        assert check.critical
        assert "FAIL" in repr(check)

    def test_failing_noncritical_check(self):
        check = PreflightCheck(
            name="Warning Check", passed=False, message="Not ideal", critical=False
        )
        assert not check.passed
        assert not check.critical
        assert "WARN" in repr(check)


class TestPreflightReport:
    """Tests for PreflightReport class."""

    def test_empty_report_passes(self):
        report = PreflightReport()
        assert report.all_passed
        assert len(report.checks) == 0

    def test_all_passing_checks(self):
        report = PreflightReport()
        report.add(PreflightCheck("Check 1", True, "OK", True))
        report.add(PreflightCheck("Check 2", True, "OK", True))
        assert report.all_passed
        assert len(report.critical_failures) == 0

    def test_critical_failure(self):
        report = PreflightReport()
        report.add(PreflightCheck("Good", True, "OK", True))
        report.add(PreflightCheck("Bad", False, "Failed", True))
        assert not report.all_passed
        assert len(report.critical_failures) == 1

    def test_noncritical_failure_still_passes(self):
        report = PreflightReport()
        report.add(PreflightCheck("Good", True, "OK", True))
        report.add(PreflightCheck("Warning", False, "Meh", False))
        assert report.all_passed
        assert len(report.warnings) == 1


class TestAPIKeyChecks:
    """Tests for API key validation."""

    def test_no_api_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing keys
            for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"]:
                os.environ.pop(key, None)

            checks = check_api_keys()
            primary_check = next(c for c in checks if c.name == "Primary API Key")
            assert not primary_check.passed

    def test_anthropic_key_present(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123"}, clear=True):
            checks = check_api_keys()
            primary_check = next(c for c in checks if c.name == "Primary API Key")
            assert primary_check.passed

    def test_openai_key_present(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=True):
            checks = check_api_keys()
            primary_check = next(c for c in checks if c.name == "Primary API Key")
            assert primary_check.passed

    def test_openrouter_fallback_warning(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            checks = check_api_keys()
            fallback_check = next(c for c in checks if c.name == "OpenRouter Fallback")
            assert not fallback_check.passed
            assert not fallback_check.critical


class TestDiskSpaceCheck:
    """Tests for disk space validation."""

    def test_sufficient_disk_space(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            check = check_disk_space(Path(tmpdir), min_gb=0.001)
            assert check.passed

    def test_insufficient_disk_space(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Require impossibly large amount
            check = check_disk_space(Path(tmpdir), min_gb=1000000)
            assert not check.passed


class TestDatabaseConnectivity:
    """Tests for database connectivity."""

    def test_nonexistent_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nonexistent.db"
            check = check_database_connectivity(db_path)
            # Nonexistent is OK (will be created)
            assert check.passed

    def test_valid_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import sqlite3

            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.close()

            check = check_database_connectivity(db_path)
            assert check.passed


class TestProtectedFilesCheck:
    """Tests for protected file validation."""

    def test_all_files_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "file1.txt").touch()
            (base / "file2.txt").touch()

            check = check_protected_files(base, ["file1.txt", "file2.txt"])
            assert check.passed

    def test_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "file1.txt").touch()

            check = check_protected_files(base, ["file1.txt", "file2.txt"])
            assert not check.passed
            assert "file2.txt" in check.message


class TestGitRepositoryCheck:
    """Tests for git repository validation."""

    def test_not_a_git_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checks = check_git_repository(Path(tmpdir))
            git_check = checks[0]
            assert not git_check.passed
            assert "Not a git repository" in git_check.message

    def test_valid_git_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import subprocess

            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)

            checks = check_git_repository(Path(tmpdir))
            git_check = checks[0]
            assert git_check.passed


class TestBackupDirectoryCheck:
    """Tests for backup directory validation."""

    def test_writable_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            check = check_backup_directory(Path(tmpdir))
            assert check.passed


class TestRunPreflightChecks:
    """Tests for the full pre-flight check suite."""

    def test_run_all_checks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Create minimal structure
            (base / "CLAUDE.md").touch()
            (base / "aragora").mkdir()
            (base / "aragora" / "__init__.py").touch()
            (base / "aragora" / "core.py").touch()
            (base / "scripts").mkdir()
            (base / "scripts" / "nomic_loop.py").touch()

            # Initialize git repo
            import subprocess

            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)

            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
                report = run_preflight_checks(base)
                # Should have multiple checks
                assert len(report.checks) > 0


class TestPhaseImports:
    """Tests that phase modules can be imported."""

    def test_import_context_phase(self):
        try:
            from scripts.nomic.phases.context import ContextPhase

            assert ContextPhase is not None
        except ImportError:
            pytest.skip("Phase modules not available")

    def test_import_debate_phase(self):
        try:
            from scripts.nomic.phases.debate import DebatePhase

            assert DebatePhase is not None
        except ImportError:
            pytest.skip("Phase modules not available")

    def test_import_design_phase(self):
        try:
            from scripts.nomic.phases.design import DesignPhase

            assert DesignPhase is not None
        except ImportError:
            pytest.skip("Phase modules not available")

    def test_import_implement_phase(self):
        try:
            from scripts.nomic.phases.implement import ImplementPhase

            assert ImplementPhase is not None
        except ImportError:
            pytest.skip("Phase modules not available")

    def test_import_verify_phase(self):
        try:
            from scripts.nomic.phases.verify import VerifyPhase

            assert VerifyPhase is not None
        except ImportError:
            pytest.skip("Phase modules not available")


class TestPhaseRecovery:
    """Tests for phase recovery mechanisms."""

    def test_import_recovery_module(self):
        try:
            from scripts.nomic.recovery import PhaseError, PhaseRecovery

            assert PhaseError is not None
            assert PhaseRecovery is not None
        except ImportError:
            pytest.skip("Recovery module not available")

    def test_phase_error_creation(self):
        try:
            from scripts.nomic.recovery import PhaseError

            error = PhaseError("test_phase", "Test error message")
            assert "test_phase" in str(error)
        except ImportError:
            pytest.skip("Recovery module not available")


class TestCircuitBreaker:
    """Tests for agent circuit breaker."""

    def test_import_circuit_breaker(self):
        try:
            from scripts.nomic.circuit_breaker import AgentCircuitBreaker

            assert AgentCircuitBreaker is not None
        except ImportError:
            pytest.skip("Circuit breaker module not available")
