"""
Tests for the server startup validation runner module.

Tests cover:
- Startup validation logs warnings
- Startup validation raises on critical issues when strict
- Skip validation environment variable
- Startup validation returns result
- Production mode is stricter
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest

from aragora.ops.deployment_validator import (
    DeploymentNotReadyError,
    Severity,
    ValidationIssue,
    ValidationResult,
)
from aragora.server.startup.validation_runner import (
    StartupValidationError,
    _is_production,
    _should_skip_validation,
    run_startup_validation,
    run_startup_validation_sync,
)


@pytest.fixture(autouse=True)
def _fresh_event_loop():
    """Ensure a fresh event loop for each test.

    ``run_startup_validation_sync()`` uses ``asyncio.run()`` internally.
    A stale or closed event loop from prior async tests causes RuntimeError.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)


class TestShouldSkipValidation:
    """Tests for _should_skip_validation function."""

    def test_skip_when_env_is_1(self, monkeypatch):
        """Skip validation when ARAGORA_SKIP_STARTUP_VALIDATION=1."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "1")
        assert _should_skip_validation() is True

    def test_skip_when_env_is_true(self, monkeypatch):
        """Skip validation when ARAGORA_SKIP_STARTUP_VALIDATION=true."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "true")
        assert _should_skip_validation() is True

    def test_skip_when_env_is_yes(self, monkeypatch):
        """Skip validation when ARAGORA_SKIP_STARTUP_VALIDATION=yes."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "yes")
        assert _should_skip_validation() is True

    def test_skip_when_env_is_TRUE_uppercase(self, monkeypatch):
        """Skip validation when ARAGORA_SKIP_STARTUP_VALIDATION=TRUE (case insensitive)."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "TRUE")
        assert _should_skip_validation() is True

    def test_skip_when_env_is_skip(self, monkeypatch):
        """Skip validation when ARAGORA_SKIP_STARTUP_VALIDATION=skip."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "skip")
        assert _should_skip_validation() is True

    def test_no_skip_when_env_is_false(self, monkeypatch):
        """Do not skip validation when ARAGORA_SKIP_STARTUP_VALIDATION=false."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "false")
        assert _should_skip_validation() is False

    def test_no_skip_when_env_is_0(self, monkeypatch):
        """Do not skip validation when ARAGORA_SKIP_STARTUP_VALIDATION=0."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "0")
        assert _should_skip_validation() is False

    def test_no_skip_when_env_not_set(self, monkeypatch):
        """Do not skip validation when ARAGORA_SKIP_STARTUP_VALIDATION is not set."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        assert _should_skip_validation() is False

    def test_no_skip_when_env_is_empty(self, monkeypatch):
        """Do not skip validation when ARAGORA_SKIP_STARTUP_VALIDATION is empty."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "")
        assert _should_skip_validation() is False


class TestIsProduction:
    """Tests for _is_production function."""

    def test_production_when_env_is_production(self, monkeypatch):
        """Is production when ARAGORA_ENV=production."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        assert _is_production() is True

    def test_production_when_env_is_prod(self, monkeypatch):
        """Is production when ARAGORA_ENV=prod."""
        monkeypatch.setenv("ARAGORA_ENV", "prod")
        assert _is_production() is True

    def test_production_when_env_is_live(self, monkeypatch):
        """Is production when ARAGORA_ENV=live."""
        monkeypatch.setenv("ARAGORA_ENV", "live")
        assert _is_production() is True

    def test_production_when_env_is_PRODUCTION_uppercase(self, monkeypatch):
        """Is production when ARAGORA_ENV=PRODUCTION (case insensitive)."""
        monkeypatch.setenv("ARAGORA_ENV", "PRODUCTION")
        assert _is_production() is True

    def test_not_production_when_env_is_development(self, monkeypatch):
        """Not production when ARAGORA_ENV=development."""
        monkeypatch.setenv("ARAGORA_ENV", "development")
        assert _is_production() is False

    def test_not_production_when_env_is_staging(self, monkeypatch):
        """Not production when ARAGORA_ENV=staging."""
        monkeypatch.setenv("ARAGORA_ENV", "staging")
        assert _is_production() is False

    def test_not_production_when_env_not_set(self, monkeypatch):
        """Not production when ARAGORA_ENV is not set."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        assert _is_production() is False

    def test_not_production_when_env_is_empty(self, monkeypatch):
        """Not production when ARAGORA_ENV is empty."""
        monkeypatch.setenv("ARAGORA_ENV", "")
        assert _is_production() is False


class TestSkipValidationEnvVar:
    """Tests for skip validation environment variable behavior."""

    @pytest.mark.asyncio
    async def test_skip_validation_env_var(self, monkeypatch, caplog):
        """Validation is skipped when ARAGORA_SKIP_STARTUP_VALIDATION=true."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "true")

        with caplog.at_level(logging.INFO):
            result = await run_startup_validation()

        assert result is None
        assert "Skipped" in caplog.text

    @pytest.mark.asyncio
    async def test_skip_validation_logs_info(self, monkeypatch, caplog):
        """Skipping validation logs an info message."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "1")

        with caplog.at_level(logging.INFO):
            await run_startup_validation()

        assert "ARAGORA_SKIP_STARTUP_VALIDATION=true" in caplog.text


class TestStartupValidationReturnsResult:
    """Tests for startup validation returning results."""

    @pytest.mark.asyncio
    async def test_startup_validation_returns_result(self, monkeypatch):
        """Startup validation returns ValidationResult when not skipped."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_result = ValidationResult(
            ready=True,
            live=True,
            issues=[],
        )

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await run_startup_validation()

        assert result is not None
        assert isinstance(result, ValidationResult)
        assert result.ready is True
        assert result.live is True

    @pytest.mark.asyncio
    async def test_startup_validation_returns_result_with_issues(self, monkeypatch):
        """Startup validation returns result including any issues."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_result = ValidationResult(
            ready=True,
            live=True,
            issues=[
                ValidationIssue(
                    component="test",
                    message="Test warning",
                    severity=Severity.WARNING,
                )
            ],
        )

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await run_startup_validation()

        assert result is not None
        assert len(result.issues) == 1
        assert result.issues[0].message == "Test warning"


class TestStartupValidationLogsWarnings:
    """Tests for startup validation logging warnings."""

    @pytest.mark.asyncio
    async def test_startup_validation_logs_warnings(self, monkeypatch, caplog):
        """Startup validation logs warning-level issues."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_result = ValidationResult(
            ready=True,
            live=True,
            issues=[
                ValidationIssue(
                    component="test_component",
                    message="This is a warning message",
                    severity=Severity.WARNING,
                )
            ],
        )

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            with caplog.at_level(logging.WARNING):
                await run_startup_validation()

        assert "test_component" in caplog.text
        assert "This is a warning message" in caplog.text

    @pytest.mark.asyncio
    async def test_startup_validation_logs_multiple_warnings(self, monkeypatch, caplog):
        """Startup validation logs all warning-level issues."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_result = ValidationResult(
            ready=True,
            live=True,
            issues=[
                ValidationIssue(
                    component="component_a",
                    message="Warning A",
                    severity=Severity.WARNING,
                ),
                ValidationIssue(
                    component="component_b",
                    message="Warning B",
                    severity=Severity.WARNING,
                ),
            ],
        )

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            with caplog.at_level(logging.WARNING):
                await run_startup_validation()

        assert "component_a" in caplog.text
        assert "Warning A" in caplog.text
        assert "component_b" in caplog.text
        assert "Warning B" in caplog.text

    @pytest.mark.asyncio
    async def test_startup_validation_logs_critical_issues(self, monkeypatch, caplog):
        """Startup validation logs critical-level issues."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_result = ValidationResult(
            ready=False,
            live=True,
            issues=[
                ValidationIssue(
                    component="critical_component",
                    message="Critical issue found",
                    severity=Severity.CRITICAL,
                )
            ],
        )

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            with caplog.at_level(logging.ERROR):
                await run_startup_validation(strict=False)

        assert "critical_component" in caplog.text
        assert "Critical issue found" in caplog.text


class TestStartupValidationRaisesOnCriticalWhenStrict:
    """Tests for startup validation raising on critical issues when strict."""

    @pytest.mark.asyncio
    async def test_startup_validation_raises_on_critical_when_strict(self, monkeypatch):
        """Startup validation raises StartupValidationError when strict and critical issues."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_result = ValidationResult(
            ready=False,
            live=True,
            issues=[
                ValidationIssue(
                    component="critical_component",
                    message="Critical issue found",
                    severity=Severity.CRITICAL,
                )
            ],
        )

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            side_effect=DeploymentNotReadyError(mock_result),
        ):
            with pytest.raises(StartupValidationError) as exc_info:
                await run_startup_validation(strict=True)

            assert exc_info.value.result is not None
            assert exc_info.value.result.ready is False
            assert len(exc_info.value.result.issues) == 1

    @pytest.mark.asyncio
    async def test_startup_validation_no_raise_when_not_strict(self, monkeypatch):
        """Startup validation does not raise when not strict, even with critical issues."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_result = ValidationResult(
            ready=False,
            live=True,
            issues=[
                ValidationIssue(
                    component="critical_component",
                    message="Critical issue found",
                    severity=Severity.CRITICAL,
                )
            ],
        )

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            # Should not raise
            result = await run_startup_validation(strict=False)

        assert result is not None
        assert result.ready is False


class TestProductionModeIsStricter:
    """Tests for production mode being stricter by default."""

    @pytest.mark.asyncio
    async def test_production_mode_is_stricter(self, monkeypatch):
        """In production, strict=True is effective by default (when strict=None)."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.delenv("ARAGORA_STRICT_DEPLOYMENT", raising=False)

        mock_result = ValidationResult(
            ready=False,
            live=True,
            issues=[
                ValidationIssue(
                    component="critical_component",
                    message="Critical issue found",
                    severity=Severity.CRITICAL,
                )
            ],
        )

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            side_effect=DeploymentNotReadyError(mock_result),
        ) as mock_validate:
            with pytest.raises(StartupValidationError):
                # strict=None means it will be determined by environment
                await run_startup_validation(strict=None)

            # Verify that validate_deployment was called with strict=True
            mock_validate.assert_called_once_with(strict=True)

    @pytest.mark.asyncio
    async def test_production_mode_can_disable_strict(self, monkeypatch):
        """Production mode can disable strict via ARAGORA_STRICT_DEPLOYMENT=false."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("ARAGORA_STRICT_DEPLOYMENT", "false")

        mock_result = ValidationResult(ready=True, live=True, issues=[])

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_validate:
            await run_startup_validation(strict=None)

            # Should be called with strict=False due to ARAGORA_STRICT_DEPLOYMENT=false
            mock_validate.assert_called_once_with(strict=False)

    @pytest.mark.asyncio
    async def test_development_mode_default_not_strict(self, monkeypatch):
        """Development mode defaults to strict=False when strict=None."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_result = ValidationResult(ready=True, live=True, issues=[])

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_validate:
            await run_startup_validation(strict=None)

            mock_validate.assert_called_once_with(strict=False)

    @pytest.mark.asyncio
    async def test_development_mode_respects_explicit_strict_true(self, monkeypatch):
        """Development mode respects strict=True when explicitly passed."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_result = ValidationResult(ready=True, live=True, issues=[])

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_validate:
            await run_startup_validation(strict=True)

            mock_validate.assert_called_once_with(strict=True)

    @pytest.mark.asyncio
    async def test_prod_shorthand_is_production(self, monkeypatch):
        """ARAGORA_ENV=prod is also treated as production."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "prod")
        monkeypatch.delenv("ARAGORA_STRICT_DEPLOYMENT", raising=False)

        mock_result = ValidationResult(ready=True, live=True, issues=[])

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_validate:
            await run_startup_validation(strict=None)

            # 'prod' should be treated as production, so strict=True
            mock_validate.assert_called_once_with(strict=True)

    @pytest.mark.asyncio
    async def test_live_is_production(self, monkeypatch):
        """ARAGORA_ENV=live is also treated as production."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "live")
        monkeypatch.delenv("ARAGORA_STRICT_DEPLOYMENT", raising=False)

        mock_result = ValidationResult(ready=True, live=True, issues=[])

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_validate:
            await run_startup_validation(strict=None)

            mock_validate.assert_called_once_with(strict=True)


class TestSyncWrapper:
    """Tests for synchronous wrapper function."""

    def test_run_startup_validation_sync_returns_result(self, monkeypatch):
        """Sync wrapper returns result from async function."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_result = ValidationResult(ready=True, live=True, issues=[])

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = run_startup_validation_sync()

        assert result is not None
        assert result.ready is True

    def test_run_startup_validation_sync_skipped(self, monkeypatch):
        """Sync wrapper returns None when validation is skipped."""
        monkeypatch.setenv("ARAGORA_SKIP_STARTUP_VALIDATION", "true")

        result = run_startup_validation_sync()

        assert result is None

    def test_run_startup_validation_sync_strict(self, monkeypatch):
        """Sync wrapper passes strict parameter correctly."""
        monkeypatch.delenv("ARAGORA_SKIP_STARTUP_VALIDATION", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        mock_result = ValidationResult(ready=True, live=True, issues=[])

        with patch(
            "aragora.ops.deployment_validator.validate_deployment",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_validate:
            run_startup_validation_sync(strict=True)

            mock_validate.assert_called_once_with(strict=True)


class TestStartupValidationError:
    """Tests for StartupValidationError exception."""

    def test_startup_validation_error_message(self):
        """StartupValidationError contains the message."""
        error = StartupValidationError("Test error message")
        assert str(error) == "Test error message"
        assert error.result is None

    def test_startup_validation_error_with_result(self):
        """StartupValidationError can include ValidationResult."""
        mock_result = ValidationResult(
            ready=False,
            live=True,
            issues=[
                ValidationIssue(
                    component="test",
                    message="Test issue",
                    severity=Severity.CRITICAL,
                )
            ],
        )
        error = StartupValidationError("Validation failed", result=mock_result)

        assert str(error) == "Validation failed"
        assert error.result is not None
        assert error.result.ready is False
        assert len(error.result.issues) == 1
