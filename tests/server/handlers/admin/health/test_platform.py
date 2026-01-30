"""
Tests for aragora.server.handlers.admin.health.platform - Platform health checks.

Tests cover:
- startup_health() - Startup report and SLO status
- encryption_health() - Encryption service status checks
- platform_health() - Platform resilience health (circuit breakers, rate limiters, DLQ)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.health.platform import (
    encryption_health,
    platform_health,
    startup_health,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockStartupReport:
    """Mock startup report for testing."""

    success: bool = True
    total_duration_seconds: float = 15.5
    slo_seconds: float = 30.0
    slo_met: bool = True
    components_initialized: int = 10
    components_failed: int = 0
    checkpoints: list = None
    error: str | None = None

    def __post_init__(self):
        if self.checkpoints is None:
            self.checkpoints = []


@dataclass
class MockCheckpoint:
    """Mock checkpoint for testing."""

    name: str = "database"
    elapsed_seconds: float = 2.5


@dataclass
class MockEncryptionKey:
    """Mock encryption key for testing."""

    version: int = 1
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, headers: dict = None, client_address: tuple = None):
        self.headers = headers or {}
        self.client_address = client_address or ("127.0.0.1", 12345)


def get_status(result) -> int:
    """Extract status code from HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def get_body(result) -> dict:
    """Extract body as dict from HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        return json.loads(body)
    return result[0] if isinstance(result[0], dict) else json.loads(result[0])


# ===========================================================================
# Tests for startup_health()
# ===========================================================================


class TestStartupHealth:
    """Tests for startup_health() function."""

    def test_startup_health_no_report_available(self):
        """Should return unknown status when no startup report is available."""
        handler = MockHandler()

        with patch(
            "aragora.server.startup_transaction.get_last_startup_report",
            return_value=None,
        ):
            result = startup_health(handler)

        assert get_status(result) == 200
        body = get_body(result)
        assert body["status"] == "unknown"
        assert body["message"] == "No startup report available"
        assert "response_time_ms" in body

    def test_startup_health_successful_within_slo(self):
        """Should return healthy status for successful startup within SLO."""
        handler = MockHandler()
        report = MockStartupReport(
            success=True,
            total_duration_seconds=15.0,
            slo_seconds=30.0,
            slo_met=True,
            components_initialized=10,
            components_failed=0,
        )

        with patch(
            "aragora.server.startup_transaction.get_last_startup_report",
            return_value=report,
        ):
            result = startup_health(handler)

        assert get_status(result) == 200
        body = get_body(result)
        assert body["status"] == "healthy"
        assert body["startup"]["success"] is True
        assert body["startup"]["slo_met"] is True
        assert body["components"]["initialized"] == 10
        assert body["components"]["failed"] == 0

    def test_startup_health_successful_exceeded_slo(self):
        """Should return warning status when startup exceeds SLO."""
        handler = MockHandler()
        report = MockStartupReport(
            success=True,
            total_duration_seconds=45.0,
            slo_seconds=30.0,
            slo_met=False,
            components_initialized=10,
            components_failed=0,
        )

        with patch(
            "aragora.server.startup_transaction.get_last_startup_report",
            return_value=report,
        ):
            result = startup_health(handler)

        body = get_body(result)
        assert body["status"] == "warning"
        assert body["startup"]["slo_met"] is False
        assert body["startup"]["duration_seconds"] == 45.0

    def test_startup_health_with_checkpoints(self):
        """Should include checkpoint information when available."""
        handler = MockHandler()
        checkpoints = [
            MockCheckpoint(name="database", elapsed_seconds=2.5),
            MockCheckpoint(name="cache", elapsed_seconds=5.0),
        ]
        report = MockStartupReport(
            success=True,
            slo_met=True,
            checkpoints=checkpoints,
        )

        with patch(
            "aragora.server.startup_transaction.get_last_startup_report",
            return_value=report,
        ):
            result = startup_health(handler)

        body = get_body(result)
        assert body["checkpoints"] is not None
        assert len(body["checkpoints"]) == 2
        assert body["checkpoints"][0]["name"] == "database"
        assert body["checkpoints"][1]["name"] == "cache"

    def test_startup_health_failed_startup(self):
        """Should return degraded status for failed startup."""
        handler = MockHandler()
        report = MockStartupReport(
            success=False,
            components_initialized=8,
            components_failed=2,
            error="Database connection failed",
        )

        with patch(
            "aragora.server.startup_transaction.get_last_startup_report",
            return_value=report,
        ):
            result = startup_health(handler)

        body = get_body(result)
        assert body["status"] == "degraded"
        assert body["error"] == "Database connection failed"
        assert body["components"]["failed"] == 2

    def test_startup_health_module_not_installed(self):
        """Should return 503 when startup transaction module is not available."""
        handler = MockHandler()

        # Create a mock module that raises ImportError when get_last_startup_report is called
        import sys

        original_modules = sys.modules.copy()

        # Remove the module to force ImportError on import
        if "aragora.server.startup_transaction" in sys.modules:
            del sys.modules["aragora.server.startup_transaction"]

        with patch.dict(sys.modules, {"aragora.server.startup_transaction": None}):
            result = startup_health(handler)

        # Restore modules
        sys.modules.update(original_modules)

        assert get_status(result) == 503
        body = get_body(result)
        assert body["status"] == "not_available"


# ===========================================================================
# Tests for encryption_health()
# ===========================================================================


class TestEncryptionHealth:
    """Tests for encryption_health() function."""

    def test_encryption_health_all_healthy(self):
        """Should return healthy status when all encryption checks pass."""
        handler = MockHandler()
        mock_service = MagicMock()
        mock_key = MockEncryptionKey(version=1)
        mock_service.get_active_key.return_value = mock_key
        mock_service.get_active_key_id.return_value = "key-1"
        mock_service.encrypt.return_value = b"encrypted"
        mock_service.decrypt.return_value = b"encryption_health_check"

        with patch(
            "aragora.security.encryption.get_encryption_service",
            return_value=mock_service,
        ):
            with patch(
                "aragora.security.encryption.CRYPTO_AVAILABLE",
                True,
            ):
                result = encryption_health(handler)

        assert get_status(result) == 200
        body = get_body(result)
        assert body["status"] == "healthy"
        assert body["health"]["cryptography_library"]["healthy"] is True
        assert body["health"]["encryption_service"]["healthy"] is True
        assert body["health"]["active_key"]["healthy"] is True
        assert body["health"]["roundtrip_test"]["healthy"] is True

    def test_encryption_health_crypto_not_installed(self):
        """Should report error when cryptography library is not installed."""
        handler = MockHandler()

        # Create a proper mock module with CRYPTO_AVAILABLE = False
        # and a mock get_encryption_service that returns None for active_key
        mock_module = MagicMock()
        mock_module.CRYPTO_AVAILABLE = False

        mock_service = MagicMock()
        mock_service.get_active_key.return_value = None
        mock_module.get_encryption_service.return_value = mock_service

        import sys

        # Must patch at import time by replacing the module in sys.modules
        original_module = sys.modules.get("aragora.security.encryption")
        sys.modules["aragora.security.encryption"] = mock_module
        try:
            # Need to call encryption_health fresh with the new module
            # But since it imports inside the function, we need a different approach
            # Let's verify the function handles the case gracefully
            result = encryption_health(handler)
            body = get_body(result)
            # The health check should detect CRYPTO_AVAILABLE = False
            assert body["health"]["cryptography_library"]["healthy"] is False
            assert "Cryptography library not installed" in body["issues"]
        finally:
            # Restore original module
            if original_module:
                sys.modules["aragora.security.encryption"] = original_module
            elif "aragora.security.encryption" in sys.modules:
                del sys.modules["aragora.security.encryption"]

    def test_encryption_health_no_active_key(self):
        """Should report issue when no active encryption key exists."""
        handler = MockHandler()
        mock_service = MagicMock()
        mock_service.get_active_key.return_value = None

        with patch(
            "aragora.security.encryption.get_encryption_service",
            return_value=mock_service,
        ):
            with patch(
                "aragora.security.encryption.CRYPTO_AVAILABLE",
                True,
            ):
                result = encryption_health(handler)

        body = get_body(result)
        assert body["health"]["active_key"]["healthy"] is False
        assert "No active encryption key" in body["issues"]

    def test_encryption_health_key_rotation_warning(self):
        """Should warn when key is older than 90 days."""
        handler = MockHandler()
        mock_service = MagicMock()
        # Create key that's 100 days old
        old_date = datetime(2025, 10, 20, tzinfo=timezone.utc)
        mock_key = MockEncryptionKey(version=1, created_at=old_date)
        mock_service.get_active_key.return_value = mock_key
        mock_service.get_active_key_id.return_value = "key-1"
        mock_service.encrypt.return_value = b"encrypted"
        mock_service.decrypt.return_value = b"encryption_health_check"

        with patch(
            "aragora.security.encryption.get_encryption_service",
            return_value=mock_service,
        ):
            with patch(
                "aragora.security.encryption.CRYPTO_AVAILABLE",
                True,
            ):
                result = encryption_health(handler)

        body = get_body(result)
        assert body["status"] == "warning"
        assert any("Rotation recommended" in w for w in body.get("warnings", []))
        assert body["health"]["active_key"]["rotation_recommended"] is True

    def test_encryption_health_roundtrip_failure(self):
        """Should report error when encrypt/decrypt roundtrip fails."""
        handler = MockHandler()
        mock_service = MagicMock()
        mock_key = MockEncryptionKey(version=1)
        mock_service.get_active_key.return_value = mock_key
        mock_service.get_active_key_id.return_value = "key-1"
        mock_service.encrypt.return_value = b"encrypted"
        mock_service.decrypt.return_value = b"wrong_data"  # Mismatch

        with patch(
            "aragora.security.encryption.get_encryption_service",
            return_value=mock_service,
        ):
            with patch(
                "aragora.security.encryption.CRYPTO_AVAILABLE",
                True,
            ):
                result = encryption_health(handler)

        body = get_body(result)
        assert body["health"]["roundtrip_test"]["healthy"] is False

    def test_encryption_health_service_error(self):
        """Should return 503 when encryption service fails to initialize."""
        handler = MockHandler()

        with patch(
            "aragora.security.encryption.CRYPTO_AVAILABLE",
            True,
        ):
            with patch(
                "aragora.security.encryption.get_encryption_service",
                side_effect=Exception("Service initialization failed"),
            ):
                result = encryption_health(handler)

        assert get_status(result) == 503
        body = get_body(result)
        assert body["status"] == "error"
        assert body["health"]["encryption_service"]["healthy"] is False


# ===========================================================================
# Tests for platform_health()
# ===========================================================================


class TestPlatformHealth:
    """Tests for platform_health() function."""

    def test_platform_health_all_components_healthy(self):
        """Should return healthy status when all platform components are operational."""
        handler = MockHandler()

        mock_limiter = MagicMock()
        mock_limiter.rpm = 60
        mock_limiter.burst_size = 10
        mock_limiter.daily_limit = 1000

        mock_resilience = MagicMock()
        mock_resilience.get_stats.return_value = {
            "platforms_tracked": 3,
            "circuit_breakers": {},
        }

        mock_dlq = MagicMock()
        mock_dlq.get_stats.return_value = {
            "pending": 5,
            "failed": 0,
            "processed": 100,
        }

        with patch(
            "aragora.server.middleware.rate_limit.platform_limiter.PLATFORM_RATE_LIMITS",
            {"slack": {}, "discord": {}},
        ):
            with patch(
                "aragora.server.middleware.rate_limit.platform_limiter.get_platform_rate_limiter",
                return_value=mock_limiter,
            ):
                with patch(
                    "aragora.integrations.platform_resilience.get_platform_resilience",
                    return_value=mock_resilience,
                ):
                    with patch(
                        "aragora.integrations.platform_resilience.DLQ_ENABLED",
                        True,
                    ):
                        with patch(
                            "aragora.integrations.platform_resilience.get_dlq",
                            return_value=mock_dlq,
                        ):
                            with patch(
                                "aragora.resilience.get_circuit_breaker",
                                return_value=None,
                            ):
                                result = platform_health(handler)

        assert get_status(result) == 200
        body = get_body(result)
        assert body["status"] in ("healthy", "healthy_with_warnings", "not_configured")
        assert "components" in body

    def test_platform_health_dlq_backing_up_warning(self):
        """Should warn when DLQ has many pending messages."""
        handler = MockHandler()

        mock_dlq = MagicMock()
        mock_dlq.get_stats.return_value = {
            "pending": 150,  # Over threshold
            "failed": 0,
            "processed": 100,
        }

        with patch(
            "aragora.server.middleware.rate_limit.platform_limiter.PLATFORM_RATE_LIMITS",
            {},
        ):
            with patch(
                "aragora.integrations.platform_resilience.get_platform_resilience",
                side_effect=ImportError,
            ):
                with patch(
                    "aragora.integrations.platform_resilience.get_dlq",
                    return_value=mock_dlq,
                ):
                    with patch(
                        "aragora.resilience.get_circuit_breaker",
                        return_value=None,
                    ):
                        result = platform_health(handler)

        body = get_body(result)
        assert body["warnings"] is not None
        assert any("150 pending" in w for w in body["warnings"])

    def test_platform_health_circuit_breaker_open(self):
        """Should report degraded status when circuit breakers are open."""
        handler = MockHandler()

        mock_cb = MagicMock()
        mock_cb.state = MagicMock(value="open")
        mock_cb.failure_count = 5
        mock_cb.success_count = 0

        with patch(
            "aragora.server.middleware.rate_limit.platform_limiter.PLATFORM_RATE_LIMITS",
            {},
        ):
            with patch(
                "aragora.integrations.platform_resilience.get_platform_resilience",
                side_effect=ImportError,
            ):
                with patch(
                    "aragora.integrations.platform_resilience.get_dlq",
                    side_effect=ImportError,
                ):
                    with patch(
                        "aragora.resilience.get_circuit_breaker",
                        return_value=mock_cb,
                    ):
                        result = platform_health(handler)

        body = get_body(result)
        assert body["status"] == "degraded"
        assert body["warnings"] is not None
        assert any("Open circuit breakers" in w for w in body["warnings"])

    def test_platform_health_modules_not_available(self):
        """Should handle gracefully when optional modules are not installed."""
        handler = MockHandler()

        # Test without patching - let it use the real imports which may fail gracefully
        result = platform_health(handler)

        assert get_status(result) == 200
        body = get_body(result)
        # Should still return a response, not crash
        assert "status" in body
        assert "components" in body

    def test_platform_health_includes_response_time(self):
        """Should include response time in the result."""
        handler = MockHandler()

        result = platform_health(handler)

        body = get_body(result)
        assert "response_time_ms" in body
        assert isinstance(body["response_time_ms"], (int, float))
        assert body["response_time_ms"] >= 0

    def test_platform_health_includes_timestamp(self):
        """Should include timestamp in the result."""
        handler = MockHandler()

        result = platform_health(handler)

        body = get_body(result)
        assert "timestamp" in body
        # Should be ISO format
        assert "T" in body["timestamp"]


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestPlatformHealthIntegration:
    """Integration tests for platform health checks."""

    def test_all_health_checks_return_valid_json(self):
        """All health check functions should return valid JSON responses."""
        handler = MockHandler()

        # Startup health - test with mock that returns None
        with patch(
            "aragora.server.startup_transaction.get_last_startup_report",
            return_value=None,
        ):
            result = startup_health(handler)
            body = get_body(result)
            assert isinstance(body, dict)
            assert "status" in body

        # Encryption health - test basic response structure
        result = encryption_health(handler)
        body = get_body(result)
        assert isinstance(body, dict)
        assert "status" in body

        # Platform health - test basic response structure
        result = platform_health(handler)
        body = get_body(result)
        assert isinstance(body, dict)
        assert "status" in body
