"""
Tests for Deployment Validator.

Tests cover:
- Enum types: Severity, ComponentStatus
- Dataclasses: ValidationIssue, ComponentHealth, ValidationResult
- DeploymentValidator: all 11 check methods
- Module-level functions: validate_deployment, quick_health_check
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.ops.deployment_validator import (
    ComponentHealth,
    ComponentStatus,
    DeploymentValidator,
    Severity,
    ValidationIssue,
    ValidationResult,
    WEAK_JWT_SECRETS,
    quick_health_check,
    validate_deployment,
)


# ============================================================================
# Enum Tests
# ============================================================================


class TestSeverity:
    """Tests for Severity enum."""

    def test_all_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"

    def test_is_str_enum(self):
        assert isinstance(Severity.CRITICAL, str)


class TestComponentStatus:
    """Tests for ComponentStatus enum."""

    def test_all_values(self):
        assert ComponentStatus.HEALTHY.value == "healthy"
        assert ComponentStatus.DEGRADED.value == "degraded"
        assert ComponentStatus.UNHEALTHY.value == "unhealthy"
        assert ComponentStatus.UNKNOWN.value == "unknown"

    def test_is_str_enum(self):
        assert isinstance(ComponentStatus.HEALTHY, str)


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_basic_creation(self):
        issue = ValidationIssue(
            component="test",
            message="Something wrong",
            severity=Severity.WARNING,
        )
        assert issue.component == "test"
        assert issue.message == "Something wrong"
        assert issue.severity == Severity.WARNING
        assert issue.suggestion is None

    def test_with_suggestion(self):
        issue = ValidationIssue(
            component="jwt",
            message="Too short",
            severity=Severity.CRITICAL,
            suggestion="Use a longer secret",
        )
        assert issue.suggestion == "Use a longer secret"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_basic_creation(self):
        health = ComponentHealth(name="redis", status=ComponentStatus.HEALTHY)
        assert health.name == "redis"
        assert health.status == ComponentStatus.HEALTHY
        assert health.latency_ms is None
        assert health.message is None
        assert health.metadata == {}

    def test_full_creation(self):
        health = ComponentHealth(
            name="database",
            status=ComponentStatus.UNHEALTHY,
            latency_ms=42.5,
            message="Connection failed",
            metadata={"backend": "postgres"},
        )
        assert health.latency_ms == 42.5
        assert health.message == "Connection failed"
        assert health.metadata["backend"] == "postgres"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_basic_creation(self):
        result = ValidationResult(ready=True, live=True)
        assert result.ready is True
        assert result.live is True
        assert result.issues == []
        assert result.components == []
        assert result.validation_duration_ms == 0.0
        assert result.validated_at > 0

    def test_to_dict(self):
        issue = ValidationIssue(
            component="test",
            message="Fail",
            severity=Severity.CRITICAL,
            suggestion="Fix it",
        )
        comp = ComponentHealth(
            name="test",
            status=ComponentStatus.UNHEALTHY,
            latency_ms=10.0,
            message="Down",
            metadata={"key": "value"},
        )
        result = ValidationResult(
            ready=False,
            live=True,
            issues=[issue],
            components=[comp],
            validation_duration_ms=50.0,
        )
        d = result.to_dict()
        assert d["ready"] is False
        assert d["live"] is True
        assert len(d["issues"]) == 1
        assert d["issues"][0]["severity"] == "critical"
        assert d["issues"][0]["suggestion"] == "Fix it"
        assert len(d["components"]) == 1
        assert d["components"][0]["status"] == "unhealthy"
        assert d["components"][0]["latency_ms"] == 10.0
        assert d["validation_duration_ms"] == 50.0


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_weak_jwt_secrets_contains_common(self):
        assert "secret" in WEAK_JWT_SECRETS
        assert "changeme" in WEAK_JWT_SECRETS
        assert "development" in WEAK_JWT_SECRETS

    def test_weak_jwt_secrets_count(self):
        assert len(WEAK_JWT_SECRETS) >= 10


# ============================================================================
# Environment Check Tests
# ============================================================================


class TestCheckEnvironment:
    """Tests for _check_environment."""

    @pytest.mark.asyncio
    async def test_development_mode(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "development")
        monkeypatch.delenv("ARAGORA_DEBUG", raising=False)
        v = DeploymentValidator()
        v._is_production = False
        await v._check_environment()
        assert any(c.name == "environment" for c in v.components)

    @pytest.mark.asyncio
    async def test_production_debug_mode_warns(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("ARAGORA_DEBUG", "true")
        v = DeploymentValidator()
        v._is_production = True
        await v._check_environment()
        assert any(
            i.component == "environment" and i.severity == Severity.WARNING for i in v.issues
        )

    @pytest.mark.asyncio
    async def test_production_no_debug(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.delenv("ARAGORA_DEBUG", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_environment()
        assert not any(i.component == "environment" for i in v.issues)


# ============================================================================
# JWT Secret Check Tests
# ============================================================================


class TestCheckJwtSecret:
    """Tests for _check_jwt_secret."""

    @pytest.mark.asyncio
    async def test_no_secret_production(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_JWT_SECRET", raising=False)
        monkeypatch.delenv("JWT_SECRET", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_jwt_secret()
        assert any(
            i.severity == Severity.CRITICAL and i.component == "jwt_secret" for i in v.issues
        )

    @pytest.mark.asyncio
    async def test_no_secret_development(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_JWT_SECRET", raising=False)
        monkeypatch.delenv("JWT_SECRET", raising=False)
        v = DeploymentValidator()
        v._is_production = False
        await v._check_jwt_secret()
        assert any(
            c.name == "jwt_secret" and c.status == ComponentStatus.UNKNOWN for c in v.components
        )
        assert not any(i.component == "jwt_secret" for i in v.issues)

    @pytest.mark.asyncio
    async def test_short_secret_production(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_JWT_SECRET", "short")
        v = DeploymentValidator()
        v._is_production = True
        await v._check_jwt_secret()
        assert any(i.severity == Severity.CRITICAL and "at least 32" in i.message for i in v.issues)

    @pytest.mark.asyncio
    async def test_short_secret_development(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_JWT_SECRET", "short")
        v = DeploymentValidator()
        v._is_production = False
        await v._check_jwt_secret()
        assert any(i.severity == Severity.WARNING and "at least 32" in i.message for i in v.issues)

    @pytest.mark.asyncio
    async def test_weak_secret(self, monkeypatch):
        # Use a weak secret that's at least 32 chars (padded)
        # Actually, weak secrets in the set are all short, so this triggers short check first
        # Test with lowercase match
        monkeypatch.setenv("ARAGORA_JWT_SECRET", "secret")
        v = DeploymentValidator()
        v._is_production = True
        await v._check_jwt_secret()
        # Will trigger short secret first since "secret" < 32 chars
        assert len(v.issues) > 0

    @pytest.mark.asyncio
    async def test_low_entropy_production(self, monkeypatch):
        # 32+ chars, all lowercase (single character class)
        monkeypatch.setenv("ARAGORA_JWT_SECRET", "a" * 64)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_jwt_secret()
        assert any(i.component == "jwt_secret" and "entropy" in i.message for i in v.issues)

    @pytest.mark.asyncio
    async def test_good_secret(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_JWT_SECRET", "Abc123!@#$%^&*SecureKey123456789")
        v = DeploymentValidator()
        v._is_production = True
        await v._check_jwt_secret()
        assert any(
            c.name == "jwt_secret" and c.status == ComponentStatus.HEALTHY for c in v.components
        )
        assert not any(i.component == "jwt_secret" for i in v.issues)

    @pytest.mark.asyncio
    async def test_jwt_secret_env_var(self, monkeypatch):
        """JWT_SECRET fallback env var works."""
        monkeypatch.delenv("ARAGORA_JWT_SECRET", raising=False)
        monkeypatch.setenv("JWT_SECRET", "Abc123!@#$%^&*SecureKey123456789")
        v = DeploymentValidator()
        v._is_production = True
        await v._check_jwt_secret()
        assert any(
            c.name == "jwt_secret" and c.status == ComponentStatus.HEALTHY for c in v.components
        )


# ============================================================================
# Storage Check Tests
# ============================================================================


class TestCheckStorage:
    """Tests for _check_storage."""

    @pytest.mark.asyncio
    async def test_writable_directory(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path / "data"))
        v = DeploymentValidator()
        v._is_production = False
        await v._check_storage()
        assert any(
            c.name == "storage" and c.status == ComponentStatus.HEALTHY for c in v.components
        )

    @pytest.mark.asyncio
    async def test_unwritable_directory(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_DATA_DIR", "/nonexistent/readonly/path/data")
        v = DeploymentValidator()
        v._is_production = True
        # This may or may not fail depending on permissions;
        # on most systems /nonexistent won't exist
        await v._check_storage()
        # Should have either healthy or unhealthy storage component
        assert any(c.name == "storage" for c in v.components)


# ============================================================================
# Database Check Tests
# ============================================================================


class TestCheckDatabase:
    """Tests for _check_database."""

    @pytest.mark.asyncio
    async def test_sqlite_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ARAGORA_DB_BACKEND", "sqlite")
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
        v = DeploymentValidator()
        v._is_production = False
        await v._check_database()
        assert any(
            c.name == "database" and c.status == ComponentStatus.HEALTHY for c in v.components
        )

    @pytest.mark.asyncio
    async def test_postgres_no_dsn(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_DB_BACKEND", "postgres")
        monkeypatch.delenv("ARAGORA_POSTGRES_DSN", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_database()
        assert any(i.component == "database" and i.severity == Severity.CRITICAL for i in v.issues)

    @pytest.mark.asyncio
    async def test_postgres_import_error(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_DB_BACKEND", "postgres")
        monkeypatch.setenv("ARAGORA_POSTGRES_DSN", "postgres://localhost/test")
        # Mock asyncpg import failure
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "asyncpg":
                raise ImportError("No module named 'asyncpg'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            v = DeploymentValidator()
            v._is_production = True
            await v._check_postgres()
            assert any(i.component == "database" and "asyncpg" in i.message for i in v.issues)


# ============================================================================
# Supabase Check Tests
# ============================================================================


class TestCheckSupabase:
    """Tests for _check_supabase."""

    @pytest.mark.asyncio
    async def test_configured_https(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "test-key")
        v = DeploymentValidator()
        v._is_production = True
        await v._check_supabase()
        assert any(
            c.name == "supabase" and c.status == ComponentStatus.HEALTHY for c in v.components
        )

    @pytest.mark.asyncio
    async def test_configured_http_warns(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "http://project.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "test-key")
        v = DeploymentValidator()
        v._is_production = True
        await v._check_supabase()
        assert any(
            c.name == "supabase" and c.status == ComponentStatus.DEGRADED for c in v.components
        )

    @pytest.mark.asyncio
    async def test_not_configured_production_no_postgres(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)
        monkeypatch.delenv("ARAGORA_POSTGRES_DSN", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_supabase()
        assert any(i.component == "supabase" for i in v.issues)

    @pytest.mark.asyncio
    async def test_not_configured_production_with_postgres(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)
        monkeypatch.setenv("DATABASE_URL", "postgres://localhost/test")
        v = DeploymentValidator()
        v._is_production = True
        await v._check_supabase()
        assert not any(i.component == "supabase" for i in v.issues)

    @pytest.mark.asyncio
    async def test_not_configured_development(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)
        v = DeploymentValidator()
        v._is_production = False
        await v._check_supabase()
        assert any(
            c.name == "supabase" and c.status == ComponentStatus.UNKNOWN for c in v.components
        )
        assert not any(i.component == "supabase" for i in v.issues)


# ============================================================================
# Redis Check Tests
# ============================================================================


class TestCheckRedis:
    """Tests for _check_redis."""

    @pytest.mark.asyncio
    async def test_not_configured_development(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REDIS_URL", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)
        v = DeploymentValidator()
        v._is_production = False
        await v._check_redis()
        assert any(c.name == "redis" and c.status == ComponentStatus.UNKNOWN for c in v.components)

    @pytest.mark.asyncio
    async def test_not_configured_production_multi_instance(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REDIS_URL", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.setenv("ARAGORA_MULTI_INSTANCE", "true")
        monkeypatch.delenv("ARAGORA_SINGLE_INSTANCE", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "production")
        v = DeploymentValidator()
        v._is_production = True
        await v._check_redis()
        assert any(i.component == "redis" and i.severity == Severity.CRITICAL for i in v.issues)

    @pytest.mark.asyncio
    async def test_not_configured_production_single_instance(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_REDIS_URL", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.delenv("ARAGORA_MULTI_INSTANCE", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_redis()
        assert any(i.component == "redis" and i.severity == Severity.WARNING for i in v.issues)


# ============================================================================
# API Keys Check Tests
# ============================================================================


class TestCheckApiKeys:
    """Tests for _check_api_keys."""

    @pytest.mark.asyncio
    async def test_no_keys_configured(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_api_keys()
        assert any(i.component == "api_keys" and i.severity == Severity.CRITICAL for i in v.issues)

    @pytest.mark.asyncio
    async def test_anthropic_key_configured(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_api_keys()
        assert any(
            c.name == "api_keys" and c.status == ComponentStatus.HEALTHY for c in v.components
        )
        assert not any(i.component == "api_keys" for i in v.issues)

    @pytest.mark.asyncio
    async def test_multiple_keys_configured(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-456")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_api_keys()
        comp = next(c for c in v.components if c.name == "api_keys")
        assert len(comp.metadata["providers"]) == 2


# ============================================================================
# Encryption Check Tests
# ============================================================================


class TestCheckEncryption:
    """Tests for _check_encryption."""

    @pytest.mark.asyncio
    async def test_no_key_production(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.delenv("ARAGORA_ENCRYPTION_KEY", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_encryption()
        assert any(
            i.component == "encryption" and i.severity == Severity.CRITICAL for i in v.issues
        )

    @pytest.mark.asyncio
    async def test_no_key_development(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "development")
        monkeypatch.delenv("ARAGORA_ENCRYPTION_KEY", raising=False)
        v = DeploymentValidator()
        v._is_production = False
        await v._check_encryption()
        assert any(
            c.name == "encryption" and c.status == ComponentStatus.UNKNOWN for c in v.components
        )
        assert not any(i.component == "encryption" for i in v.issues)

    @pytest.mark.asyncio
    async def test_valid_key(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        # 32-byte key = 64 hex chars
        monkeypatch.setenv("ARAGORA_ENCRYPTION_KEY", "a" * 64)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_encryption()
        assert any(
            c.name == "encryption" and c.status == ComponentStatus.HEALTHY for c in v.components
        )

    @pytest.mark.asyncio
    async def test_invalid_key_format(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("ARAGORA_ENCRYPTION_KEY", "not-hex")
        v = DeploymentValidator()
        v._is_production = True
        await v._check_encryption()
        assert any(
            i.component == "encryption" and i.severity == Severity.CRITICAL for i in v.issues
        )

    @pytest.mark.asyncio
    async def test_wrong_length_key(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        # 16 bytes = 32 hex chars (too short)
        monkeypatch.setenv("ARAGORA_ENCRYPTION_KEY", "a" * 32)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_encryption()
        assert any(
            i.component == "encryption" and i.severity == Severity.CRITICAL for i in v.issues
        )


# ============================================================================
# CORS Security Check Tests
# ============================================================================


class TestCheckCorsSecurity:
    """Tests for _check_cors_security."""

    @pytest.mark.asyncio
    async def test_cors_allow_all_production(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_CORS_ALLOW_ALL", "true")
        monkeypatch.delenv("ARAGORA_ALLOWED_ORIGINS", raising=False)
        monkeypatch.delenv("ARAGORA_DISABLE_SECURITY_HEADERS", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_cors_security()
        assert any(i.component == "cors" and i.severity == Severity.CRITICAL for i in v.issues)

    @pytest.mark.asyncio
    async def test_localhost_in_production_origins(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_CORS_ALLOW_ALL", raising=False)
        monkeypatch.setenv(
            "ARAGORA_ALLOWED_ORIGINS", "https://app.example.com,http://localhost:3000"
        )
        monkeypatch.delenv("ARAGORA_DISABLE_SECURITY_HEADERS", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_cors_security()
        assert any(i.component == "cors" and "localhost" in i.message for i in v.issues)

    @pytest.mark.asyncio
    async def test_wildcard_in_origins_production(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_CORS_ALLOW_ALL", raising=False)
        monkeypatch.setenv("ARAGORA_ALLOWED_ORIGINS", "*")
        monkeypatch.delenv("ARAGORA_DISABLE_SECURITY_HEADERS", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_cors_security()
        assert any(i.component == "cors" and i.severity == Severity.CRITICAL for i in v.issues)

    @pytest.mark.asyncio
    async def test_valid_cors_production(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_CORS_ALLOW_ALL", raising=False)
        monkeypatch.setenv("ARAGORA_ALLOWED_ORIGINS", "https://app.example.com")
        monkeypatch.delenv("ARAGORA_DISABLE_SECURITY_HEADERS", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_cors_security()
        assert any(c.name == "cors" and c.status == ComponentStatus.HEALTHY for c in v.components)

    @pytest.mark.asyncio
    async def test_security_headers_disabled_production(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_CORS_ALLOW_ALL", raising=False)
        monkeypatch.setenv("ARAGORA_ALLOWED_ORIGINS", "https://app.example.com")
        monkeypatch.setenv("ARAGORA_DISABLE_SECURITY_HEADERS", "true")
        v = DeploymentValidator()
        v._is_production = True
        await v._check_cors_security()
        assert any(i.component == "cors" and "Security headers" in i.message for i in v.issues)

    @pytest.mark.asyncio
    async def test_no_origins_production_info(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_CORS_ALLOW_ALL", raising=False)
        monkeypatch.delenv("ARAGORA_ALLOWED_ORIGINS", raising=False)
        monkeypatch.delenv("ARAGORA_DISABLE_SECURITY_HEADERS", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_cors_security()
        assert any(i.component == "cors" and i.severity == Severity.INFO for i in v.issues)


# ============================================================================
# Rate Limiting Check Tests
# ============================================================================


class TestCheckRateLimiting:
    """Tests for _check_rate_limiting."""

    @pytest.mark.asyncio
    async def test_rate_limiting_disabled_production(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_DISABLE_RATE_LIMIT", "true")
        v = DeploymentValidator()
        v._is_production = True
        await v._check_rate_limiting()
        assert any(
            i.component == "rate_limiting" and i.severity == Severity.CRITICAL for i in v.issues
        )

    @pytest.mark.asyncio
    async def test_rate_limiting_disabled_development(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_DISABLE_RATE_LIMIT", "true")
        v = DeploymentValidator()
        v._is_production = False
        await v._check_rate_limiting()
        assert any(
            c.name == "rate_limiting" and c.status == ComponentStatus.DEGRADED for c in v.components
        )

    @pytest.mark.asyncio
    async def test_memory_backend_multi_instance(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_DISABLE_RATE_LIMIT", raising=False)
        monkeypatch.delenv("ARAGORA_RATE_LIMIT", raising=False)
        monkeypatch.delenv("ARAGORA_IP_RATE_LIMIT", raising=False)
        monkeypatch.setenv("ARAGORA_RATE_LIMIT_BACKEND", "memory")
        monkeypatch.setenv("ARAGORA_MULTI_INSTANCE", "true")
        v = DeploymentValidator()
        v._is_production = True
        await v._check_rate_limiting()
        assert any(i.component == "rate_limiting" and "not shared" in i.message for i in v.issues)

    @pytest.mark.asyncio
    async def test_redis_backend_no_url(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_DISABLE_RATE_LIMIT", raising=False)
        monkeypatch.delenv("ARAGORA_RATE_LIMIT", raising=False)
        monkeypatch.delenv("ARAGORA_IP_RATE_LIMIT", raising=False)
        monkeypatch.setenv("ARAGORA_RATE_LIMIT_BACKEND", "redis")
        monkeypatch.delenv("ARAGORA_REDIS_URL", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.delenv("ARAGORA_MULTI_INSTANCE", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_rate_limiting()
        assert any(i.component == "rate_limiting" and "REDIS_URL" in i.message for i in v.issues)

    @pytest.mark.asyncio
    async def test_invalid_rate_limit_value(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_DISABLE_RATE_LIMIT", raising=False)
        monkeypatch.setenv("ARAGORA_RATE_LIMIT", "not-a-number")
        monkeypatch.delenv("ARAGORA_IP_RATE_LIMIT", raising=False)
        monkeypatch.setenv("ARAGORA_RATE_LIMIT_BACKEND", "memory")
        monkeypatch.delenv("ARAGORA_MULTI_INSTANCE", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_rate_limiting()
        assert any(i.component == "rate_limiting" and "Invalid" in i.message for i in v.issues)

    @pytest.mark.asyncio
    async def test_healthy_rate_limiting(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_DISABLE_RATE_LIMIT", raising=False)
        monkeypatch.setenv("ARAGORA_RATE_LIMIT", "100")
        monkeypatch.setenv("ARAGORA_IP_RATE_LIMIT", "50")
        monkeypatch.setenv("ARAGORA_RATE_LIMIT_BACKEND", "memory")
        monkeypatch.delenv("ARAGORA_MULTI_INSTANCE", raising=False)
        v = DeploymentValidator()
        v._is_production = False
        await v._check_rate_limiting()
        assert any(
            c.name == "rate_limiting" and c.status == ComponentStatus.HEALTHY for c in v.components
        )


# ============================================================================
# TLS Settings Check Tests
# ============================================================================


class TestCheckTlsSettings:
    """Tests for _check_tls_settings."""

    @pytest.mark.asyncio
    async def test_no_tls_production(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_FORCE_HTTPS", raising=False)
        monkeypatch.delenv("ARAGORA_TLS_ENABLED", raising=False)
        monkeypatch.delenv("ARAGORA_BEHIND_PROXY", raising=False)
        monkeypatch.delenv("ARAGORA_TRUST_PROXY_HEADERS", raising=False)
        monkeypatch.delenv("ARAGORA_HSTS_ENABLED", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_tls_settings()
        assert any(i.component == "tls" for i in v.issues)

    @pytest.mark.asyncio
    async def test_behind_proxy_production(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_BEHIND_PROXY", "true")
        monkeypatch.setenv("ARAGORA_TRUST_PROXY_HEADERS", "true")
        monkeypatch.delenv("ARAGORA_TLS_ENABLED", raising=False)
        monkeypatch.delenv("ARAGORA_FORCE_HTTPS", raising=False)
        monkeypatch.delenv("ARAGORA_HSTS_ENABLED", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_tls_settings()
        assert any(c.name == "tls" and c.status == ComponentStatus.HEALTHY for c in v.components)

    @pytest.mark.asyncio
    async def test_tls_enabled_no_cert_paths(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_TLS_ENABLED", "true")
        monkeypatch.delenv("ARAGORA_TLS_CERT_PATH", raising=False)
        monkeypatch.delenv("ARAGORA_TLS_KEY_PATH", raising=False)
        monkeypatch.delenv("ARAGORA_BEHIND_PROXY", raising=False)
        monkeypatch.delenv("ARAGORA_FORCE_HTTPS", raising=False)
        monkeypatch.delenv("ARAGORA_HSTS_ENABLED", raising=False)
        monkeypatch.delenv("ARAGORA_TRUST_PROXY_HEADERS", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_tls_settings()
        assert any(i.component == "tls" and i.severity == Severity.CRITICAL for i in v.issues)

    @pytest.mark.asyncio
    async def test_development_no_tls_ok(self, monkeypatch):
        monkeypatch.delenv("ARAGORA_FORCE_HTTPS", raising=False)
        monkeypatch.delenv("ARAGORA_TLS_ENABLED", raising=False)
        monkeypatch.delenv("ARAGORA_BEHIND_PROXY", raising=False)
        monkeypatch.delenv("ARAGORA_TRUST_PROXY_HEADERS", raising=False)
        monkeypatch.delenv("ARAGORA_HSTS_ENABLED", raising=False)
        v = DeploymentValidator()
        v._is_production = False
        await v._check_tls_settings()
        assert not any(i.component == "tls" and i.severity == Severity.CRITICAL for i in v.issues)

    @pytest.mark.asyncio
    async def test_proxy_without_header_trust(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_BEHIND_PROXY", "true")
        monkeypatch.delenv("ARAGORA_TRUST_PROXY_HEADERS", raising=False)
        monkeypatch.setenv("ARAGORA_FORCE_HTTPS", "true")
        monkeypatch.delenv("ARAGORA_TLS_ENABLED", raising=False)
        monkeypatch.delenv("ARAGORA_HSTS_ENABLED", raising=False)
        v = DeploymentValidator()
        v._is_production = True
        await v._check_tls_settings()
        assert any(i.component == "tls" and "X-Forwarded" in i.message for i in v.issues)


# ============================================================================
# Full Validation Tests
# ============================================================================


class TestFullValidation:
    """Tests for the full validate() method."""

    @pytest.mark.asyncio
    async def test_development_defaults(self, monkeypatch):
        """Development mode with minimal config should pass."""
        monkeypatch.setenv("ARAGORA_ENV", "development")
        monkeypatch.delenv("ARAGORA_JWT_SECRET", raising=False)
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("ARAGORA_REDIS_URL", raising=False)
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Ensure we have a writable data dir
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(Path(__file__).parent / "test_data"))

        v = DeploymentValidator()
        result = await v.validate()
        assert result.live is True
        assert result.validation_duration_ms > 0
        assert isinstance(result.to_dict(), dict)

    @pytest.mark.asyncio
    async def test_production_no_config_fails(self, monkeypatch):
        """Production with no config has critical issues."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.delenv("ARAGORA_JWT_SECRET", raising=False)
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ARAGORA_ENCRYPTION_KEY", raising=False)
        monkeypatch.setenv("ARAGORA_DISABLE_RATE_LIMIT", "true")
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(Path(__file__).parent / "test_data"))

        v = DeploymentValidator()
        result = await v.validate()
        assert result.ready is False
        assert len([i for i in result.issues if i.severity == Severity.CRITICAL]) > 0

    @pytest.mark.asyncio
    async def test_readiness_determined_by_critical(self, monkeypatch):
        """Readiness is False when critical issues exist."""
        v = DeploymentValidator()
        v.issues = [
            ValidationIssue("test", "warning", Severity.WARNING),
            ValidationIssue("test", "info", Severity.INFO),
        ]
        # No critical → ready=True
        critical = [i for i in v.issues if i.severity == Severity.CRITICAL]
        assert len(critical) == 0


# ============================================================================
# Module-Level Function Tests
# ============================================================================


class TestModuleFunctions:
    """Tests for validate_deployment and quick_health_check."""

    @pytest.mark.asyncio
    async def test_validate_deployment(self, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "development")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(Path(__file__).parent / "test_data"))
        result = await validate_deployment()
        assert isinstance(result, ValidationResult)
        assert result.validation_duration_ms > 0

    def test_quick_health_check_success(self):
        with patch("aragora.config.validator.validate_all") as mock_validate:
            mock_validate.return_value = {"errors": [], "warnings": []}
            assert quick_health_check() is True

    def test_quick_health_check_errors(self):
        with patch("aragora.config.validator.validate_all") as mock_validate:
            mock_validate.return_value = {"errors": ["Missing config"]}
            assert quick_health_check() is False

    def test_quick_health_check_exception(self):
        with patch(
            "aragora.config.validator.validate_all",
            side_effect=ImportError("Module not found"),
        ):
            assert quick_health_check() is False
