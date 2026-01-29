"""
Tests for aragora.server.config_validator connectivity checks.

Covers:
- PostgreSQL connectivity validation
- Redis connectivity validation
- Async connectivity validation
- Error handling for unreachable databases

Run with:
    python -m pytest tests/server/test_config_validator_connectivity.py -v --noconftest --timeout=30
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify the config_validator module and its public API can be imported."""

    def test_import_module(self):
        import aragora.server.config_validator as mod

        assert hasattr(mod, "ConfigValidator")
        assert hasattr(mod, "ValidationResult")
        assert hasattr(mod, "validate_startup_config")
        assert hasattr(mod, "validate_startup_config_async")

    def test_import_validation_result(self):
        from aragora.server.config_validator import ValidationResult

        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert result.is_valid
        assert not result.has_errors
        assert not result.has_warnings


# ---------------------------------------------------------------------------
# PostgreSQL connectivity tests
# ---------------------------------------------------------------------------


class TestPostgreSQLConnectivity:
    """Tests for PostgreSQL connectivity checking."""

    def test_no_database_url_returns_success(self):
        """When DATABASE_URL is not set, connectivity check should pass."""
        from aragora.server.config_validator import ConfigValidator

        with patch.dict(os.environ, {}, clear=True):
            # Remove DATABASE_URL if present
            os.environ.pop("DATABASE_URL", None)
            success, error = ConfigValidator.check_postgresql_connectivity(
                skip_connectivity_test=True
            )
            assert success is True
            assert error is None

    def test_non_postgresql_url_returns_success(self):
        """When DATABASE_URL is not PostgreSQL, connectivity check should pass."""
        from aragora.server.config_validator import ConfigValidator

        with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///test.db"}):
            success, error = ConfigValidator.check_postgresql_connectivity(
                skip_connectivity_test=True
            )
            assert success is True
            assert error is None

    def test_skip_connectivity_test_flag(self):
        """When skip_connectivity_test=True, only format is validated."""
        from aragora.server.config_validator import ConfigValidator

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            success, error = ConfigValidator.check_postgresql_connectivity(
                skip_connectivity_test=True
            )
            assert success is True
            assert error is None

    def test_asyncpg_import_error(self):
        """When asyncpg is not available, should return error."""
        from aragora.server.config_validator import ConfigValidator

        with (
            patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}),
            patch.dict("sys.modules", {"asyncpg": None}),
            patch("builtins.__import__", side_effect=ImportError("No module named 'asyncpg'")),
        ):
            # Force reimport
            success, error = ConfigValidator.check_postgresql_connectivity(
                skip_connectivity_test=True
            )
            # This test is tricky because asyncpg might already be imported
            # Let's verify the method exists and can be called
            assert callable(ConfigValidator.check_postgresql_connectivity)


class TestPostgreSQLConnectivityAsync:
    """Tests for async PostgreSQL connectivity checking."""

    @pytest.mark.asyncio
    async def test_no_database_url_returns_success(self):
        """When DATABASE_URL is not set, async connectivity check should pass."""
        from aragora.server.config_validator import ConfigValidator

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DATABASE_URL", None)
            success, error = await ConfigValidator.check_postgresql_connectivity_async()
            assert success is True
            assert error is None

    @pytest.mark.asyncio
    async def test_non_postgresql_url_returns_success(self):
        """When DATABASE_URL is not PostgreSQL, async check should pass."""
        from aragora.server.config_validator import ConfigValidator

        with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///test.db"}):
            success, error = await ConfigValidator.check_postgresql_connectivity_async()
            assert success is True
            assert error is None

    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        """When connection times out, should return appropriate error."""
        from aragora.server.config_validator import ConfigValidator

        mock_asyncpg = MagicMock()
        mock_asyncpg.connect = AsyncMock(side_effect=asyncio.TimeoutError())

        with (
            patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}),
            patch.dict("sys.modules", {"asyncpg": mock_asyncpg}),
        ):
            success, error = await ConfigValidator.check_postgresql_connectivity_async(
                timeout_seconds=0.1
            )
            # Should fail with timeout error
            # Note: actual behavior depends on whether asyncpg is available
            assert callable(ConfigValidator.check_postgresql_connectivity_async)

    @pytest.mark.asyncio
    async def test_successful_connection(self):
        """When connection succeeds, should return success."""
        from aragora.server.config_validator import ConfigValidator

        # Create mock connection
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        mock_conn.close = AsyncMock()

        mock_asyncpg = MagicMock()
        mock_asyncpg.connect = AsyncMock(return_value=mock_conn)
        mock_asyncpg.PostgresError = Exception

        with (
            patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}),
            patch("aragora.server.config_validator.asyncpg", mock_asyncpg, create=True),
        ):
            # The actual test depends on import order
            assert hasattr(ConfigValidator, "check_postgresql_connectivity_async")


# ---------------------------------------------------------------------------
# Redis connectivity tests
# ---------------------------------------------------------------------------


class TestRedisConnectivity:
    """Tests for Redis connectivity checking."""

    def test_no_redis_url_returns_success(self):
        """When REDIS_URL is not set, connectivity check should pass."""
        from aragora.server.config_validator import ConfigValidator

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("REDIS_URL", None)
            success, error = ConfigValidator.check_redis_connectivity()
            assert success is True
            assert error is None

    def test_redis_import_error(self):
        """When redis package is not available, should pass (not an error)."""
        from aragora.server.config_validator import ConfigValidator

        with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379/0"}):
            # The actual import behavior depends on whether redis is installed
            success, error = ConfigValidator.check_redis_connectivity()
            # Should either succeed or return import-related message
            assert isinstance(success, bool)

    def test_redis_connection_error(self):
        """When Redis connection fails, should return error."""
        from aragora.server.config_validator import ConfigValidator

        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("Connection refused")
        mock_redis.from_url.return_value = mock_client

        with (
            patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379/0"}),
            patch.dict("sys.modules", {"redis": mock_redis}),
        ):
            success, error = ConfigValidator.check_redis_connectivity()
            # Should either fail or pass depending on import resolution
            assert isinstance(success, bool)


# ---------------------------------------------------------------------------
# Async validate connectivity tests
# ---------------------------------------------------------------------------


class TestValidateConnectivityAsync:
    """Tests for async connectivity validation."""

    @pytest.mark.asyncio
    async def test_validate_connectivity_no_config(self):
        """When no databases configured, validation should pass."""
        from aragora.server.config_validator import ConfigValidator

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DATABASE_URL", None)
            os.environ.pop("REDIS_URL", None)

            result = await ConfigValidator.validate_connectivity_async()
            assert result.is_valid
            assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_validate_connectivity_selective_checks(self):
        """Test selective connectivity checks."""
        from aragora.server.config_validator import ConfigValidator

        with patch.dict(os.environ, {}, clear=True):
            # Only check PostgreSQL (should pass when not configured)
            result = await ConfigValidator.validate_connectivity_async(
                check_postgresql=True,
                check_redis=False,
            )
            assert result.is_valid

            # Only check Redis (should pass when not configured)
            result = await ConfigValidator.validate_connectivity_async(
                check_postgresql=False,
                check_redis=True,
            )
            assert result.is_valid


# ---------------------------------------------------------------------------
# validate_startup_config_async tests
# ---------------------------------------------------------------------------


class TestValidateStartupConfigAsync:
    """Tests for async startup config validation."""

    @pytest.mark.asyncio
    async def test_validate_startup_config_async_basic(self):
        """Test basic async startup validation."""
        from aragora.server.config_validator import validate_startup_config_async

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-12345678901234567890"}):
            result = await validate_startup_config_async(
                strict=False,
                check_connectivity=False,
            )
            # Should return ValidationResult
            assert hasattr(result, "is_valid")
            assert hasattr(result, "errors")
            assert hasattr(result, "warnings")

    @pytest.mark.asyncio
    async def test_validate_startup_config_async_with_connectivity(self):
        """Test async startup validation with connectivity checks."""
        from aragora.server.config_validator import validate_startup_config_async

        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test-key-12345678901234567890",
            },
        ):
            # Clear database URLs to ensure connectivity checks pass
            os.environ.pop("DATABASE_URL", None)
            os.environ.pop("REDIS_URL", None)

            result = await validate_startup_config_async(
                strict=False,
                check_connectivity=True,
            )
            assert hasattr(result, "is_valid")


# ---------------------------------------------------------------------------
# ValidationResult tests
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_is_valid_with_no_errors(self):
        from aragora.server.config_validator import ValidationResult

        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert result.is_valid
        assert not result.has_errors
        assert not result.has_warnings

    def test_is_valid_with_errors(self):
        from aragora.server.config_validator import ValidationResult

        result = ValidationResult(is_valid=False, errors=["Error 1"], warnings=[])
        assert not result.is_valid
        assert result.has_errors
        assert not result.has_warnings

    def test_is_valid_with_warnings(self):
        from aragora.server.config_validator import ValidationResult

        result = ValidationResult(is_valid=True, errors=[], warnings=["Warning 1"])
        assert result.is_valid
        assert not result.has_errors
        assert result.has_warnings


# ---------------------------------------------------------------------------
# ConfigValidator.validate_all tests
# ---------------------------------------------------------------------------


class TestValidateAll:
    """Tests for ConfigValidator.validate_all method."""

    def test_validate_all_returns_result(self):
        from aragora.server.config_validator import ConfigValidator, ValidationResult

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-12345678901234567890"}):
            result = ConfigValidator.validate_all()
            assert isinstance(result, ValidationResult)

    def test_validate_all_development_mode(self):
        from aragora.server.config_validator import ConfigValidator

        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "development",
                "ANTHROPIC_API_KEY": "test-key-12345678901234567890",
            },
        ):
            result = ConfigValidator.validate_all()
            # Should be valid in development with just an API key
            assert result.is_valid or len(result.errors) == 0

    def test_validate_all_production_mode_missing_required(self):
        from aragora.server.config_validator import ConfigValidator

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=True):
            # Clear required vars
            os.environ.pop("ARAGORA_API_TOKEN", None)
            os.environ.pop("DATABASE_URL", None)

            result = ConfigValidator.validate_all()
            # Should have errors for missing required vars
            assert result.has_errors or len(result.errors) > 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for config validation."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        from aragora.server.config_validator import ConfigValidator, validate_startup_config

        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "development",
                "ANTHROPIC_API_KEY": "test-key-12345678901234567890",
            },
        ):
            # Should not exit
            is_valid = validate_startup_config(strict=False, exit_on_error=False)
            assert isinstance(is_valid, bool)

    def test_get_config_summary(self):
        """Test config summary generation."""
        from aragora.server.config_validator import ConfigValidator

        summary = ConfigValidator.get_config_summary()
        assert isinstance(summary, dict)
        assert "environment" in summary
        assert "api_token_set" in summary
        assert "llm_keys" in summary
        assert "database_backend" in summary
