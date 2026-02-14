"""
Tests for aragora.server.startup.validation_runner module.

Tests startup deployment validation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.startup.validation_runner import (
    StartupValidationError,
    _is_production,
    _should_skip_validation,
)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestShouldSkipValidation:
    """Tests for _should_skip_validation function."""

    def test_skip_with_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test skip validation with ARAGORA_SKIP_STARTUP_VALIDATION=true."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "true")
        assert _should_skip_validation() is True

    def test_skip_with_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test skip validation with ARAGORA_SKIP_STARTUP_VALIDATION=1."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "1")
        assert _should_skip_validation() is True

    def test_skip_with_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test skip validation with ARAGORA_SKIP_STARTUP_VALIDATION=yes."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "yes")
        assert _should_skip_validation() is True

    def test_skip_with_skip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test skip validation with ARAGORA_SKIP_STARTUP_VALIDATION=skip."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "skip")
        assert _should_skip_validation() is True

    def test_no_skip_with_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test no skip with ARAGORA_SKIP_STARTUP_VALIDATION=false."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "false")
        assert _should_skip_validation() is False

    def test_no_skip_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test no skip when env var not set."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        assert _should_skip_validation() is False


class TestIsProduction:
    """Tests for _is_production function."""

    def test_production_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test production detection."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        assert _is_production() is True

    def test_prod_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test prod detection."""
        monkeypatch.setenv("ARAGORA_ENV", "prod")
        assert _is_production() is True

    def test_live_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test live detection."""
        monkeypatch.setenv("ARAGORA_ENV", "live")
        assert _is_production() is True

    def test_development_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test development is not production."""
        monkeypatch.setenv("ARAGORA_ENV", "development")
        assert _is_production() is False

    def test_default_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default is development."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        assert _is_production() is False


# =============================================================================
# StartupValidationError Tests
# =============================================================================


class TestStartupValidationError:
    """Tests for StartupValidationError exception."""

    def test_error_message(self) -> None:
        """Test error message is set correctly."""
        error = StartupValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert error.result is None

    def test_error_with_result(self) -> None:
        """Test error with result attached."""
        mock_result = MagicMock()
        error = StartupValidationError("Validation failed", result=mock_result)
        assert error.result == mock_result


# =============================================================================
# run_startup_validation Tests
# =============================================================================


class TestRunStartupValidation:
    """Tests for run_startup_validation function."""

    @pytest.mark.asyncio
    async def test_skip_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test validation skipped via environment variable."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "true")

        from aragora.server.startup.validation_runner import run_startup_validation

        result = await run_startup_validation()
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful validation returns result."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_result = MagicMock()
        mock_result.ready = True
        mock_result.issues = []
        mock_result.validation_duration_ms = 50.0

        mock_severity = MagicMock()
        mock_severity.CRITICAL = "critical"
        mock_severity.WARNING = "warning"
        mock_severity.INFO = "info"

        mock_validator = MagicMock()
        mock_validator.validate_deployment = AsyncMock(return_value=mock_result)
        mock_validator.Severity = mock_severity
        mock_validator.DeploymentNotReadyError = Exception

        with patch.dict("sys.modules", {"aragora.ops.deployment_validator": mock_validator}):
            from aragora.server.startup.validation_runner import run_startup_validation

            result = await run_startup_validation(strict=False)

        assert result == mock_result

    @pytest.mark.asyncio
    async def test_validation_with_warnings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test validation with warnings passes but logs them."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_warning = MagicMock()
        mock_warning.severity = "warning"
        mock_warning.component = "test"
        mock_warning.message = "Test warning"
        mock_warning.suggestion = "Fix it"

        mock_result = MagicMock()
        mock_result.ready = True
        mock_result.issues = [mock_warning]
        mock_result.validation_duration_ms = 50.0

        mock_severity = MagicMock()
        mock_severity.CRITICAL = "critical"
        mock_severity.WARNING = "warning"
        mock_severity.INFO = "info"

        mock_validator = MagicMock()
        mock_validator.validate_deployment = AsyncMock(return_value=mock_result)
        mock_validator.Severity = mock_severity
        mock_validator.DeploymentNotReadyError = Exception

        with patch.dict("sys.modules", {"aragora.ops.deployment_validator": mock_validator}):
            from aragora.server.startup.validation_runner import run_startup_validation

            result = await run_startup_validation(strict=False)

        assert result == mock_result

    @pytest.mark.asyncio
    async def test_strict_validation_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test strict validation raises on critical issues."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_result = MagicMock()

        class MockDeploymentNotReadyError(Exception):
            def __init__(self, msg: str) -> None:
                super().__init__(msg)
                self.result = mock_result

        mock_validator = MagicMock()
        mock_validator.validate_deployment = AsyncMock(
            side_effect=MockDeploymentNotReadyError("Critical issues found")
        )
        mock_validator.Severity = MagicMock()
        mock_validator.DeploymentNotReadyError = MockDeploymentNotReadyError

        import importlib
        import aragora.server.startup.validation_runner as vr_module

        with patch.dict("sys.modules", {"aragora.ops.deployment_validator": mock_validator}):
            importlib.reload(vr_module)

            with pytest.raises(StartupValidationError) as exc_info:
                await vr_module.run_startup_validation(strict=True)

        # Restore original module
        importlib.reload(vr_module)

        assert "Critical issues found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_production_defaults_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test production environment defaults to strict=True."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.delenv("ARAGORA_STRICT_DEPLOYMENT", raising=False)

        mock_result = MagicMock()
        mock_result.ready = True
        mock_result.issues = []
        mock_result.validation_duration_ms = 50.0

        mock_severity = MagicMock()
        mock_severity.CRITICAL = "critical"
        mock_severity.WARNING = "warning"
        mock_severity.INFO = "info"

        mock_validator = MagicMock()
        mock_validator.validate_deployment = AsyncMock(return_value=mock_result)
        mock_validator.Severity = mock_severity
        mock_validator.DeploymentNotReadyError = Exception

        with patch.dict("sys.modules", {"aragora.ops.deployment_validator": mock_validator}):
            from aragora.server.startup.validation_runner import run_startup_validation

            result = await run_startup_validation()

        # Should have called with strict=True
        mock_validator.validate_deployment.assert_awaited_once_with(strict=True)

    @pytest.mark.asyncio
    async def test_production_strict_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ARAGORA_STRICT_DEPLOYMENT=false disables strict in production."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("ARAGORA_STRICT_DEPLOYMENT", "false")

        mock_result = MagicMock()
        mock_result.ready = True
        mock_result.issues = []
        mock_result.validation_duration_ms = 50.0

        mock_severity = MagicMock()
        mock_severity.CRITICAL = "critical"
        mock_severity.WARNING = "warning"
        mock_severity.INFO = "info"

        mock_validator = MagicMock()
        mock_validator.validate_deployment = AsyncMock(return_value=mock_result)
        mock_validator.Severity = mock_severity
        mock_validator.DeploymentNotReadyError = Exception

        with patch.dict("sys.modules", {"aragora.ops.deployment_validator": mock_validator}):
            from aragora.server.startup.validation_runner import run_startup_validation

            result = await run_startup_validation()

        # Should have called with strict=False
        mock_validator.validate_deployment.assert_awaited_once_with(strict=False)

    @pytest.mark.asyncio
    async def test_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ImportError returns None."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)

        with patch.dict("sys.modules", {"aragora.ops.deployment_validator": None}):
            import importlib
            import aragora.server.startup.validation_runner as vr_module

            importlib.reload(vr_module)
            result = await vr_module.run_startup_validation()

        assert result is None


# =============================================================================
# run_startup_validation_sync Tests
# =============================================================================


class TestRunStartupValidationSync:
    """Tests for run_startup_validation_sync function."""

    def test_skip_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test sync validation skipped via environment variable."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "true")

        from aragora.server.startup.validation_runner import run_startup_validation_sync

        result = run_startup_validation_sync()
        assert result is None
