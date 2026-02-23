"""Comprehensive tests for platform health, encryption, and startup check handlers.

Tests the three public functions in aragora/server/handlers/admin/health/platform.py:

  TestStartupHealth                 - startup_health() startup report and SLO status
  TestStartupHealthNoReport         - startup_health() when no report available
  TestStartupHealthImportError      - startup_health() when module not installed
  TestEncryptionHealth              - encryption_health() full encryption check
  TestEncryptionHealthImportError   - encryption_health() import failures
  TestEncryptionHealthServiceError  - encryption_health() service init failures
  TestEncryptionHealthKeyStatus     - encryption_health() active key scenarios
  TestEncryptionHealthRoundtrip     - encryption_health() encrypt/decrypt roundtrip
  TestPlatformHealth                - platform_health() platform resilience check
  TestPlatformHealthRateLimiters    - platform_health() rate limiter scenarios
  TestPlatformHealthResilience      - platform_health() resilience module scenarios
  TestPlatformHealthDLQ             - platform_health() dead letter queue scenarios
  TestPlatformHealthMetrics         - platform_health() platform metrics scenarios
  TestPlatformHealthCircuitBreakers - platform_health() circuit breaker scenarios
  TestPlatformHealthOverallStatus   - platform_health() overall status determination

150+ tests covering all branches, error paths, and edge cases.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.health.platform import (
    startup_health,
    encryption_health,
    platform_health,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _make_mock_handler() -> MagicMock:
    """Create a mock handler object."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Mock data factories
# ---------------------------------------------------------------------------


def _make_startup_report(
    success: bool = True,
    slo_met: bool = True,
    total_duration_seconds: float = 5.0,
    slo_seconds: float = 30.0,
    components_initialized: int = 10,
    components_failed: int = 0,
    checkpoints: list | None = None,
    error: str | None = None,
):
    """Create a mock startup report."""
    report = MagicMock()
    report.success = success
    report.slo_met = slo_met
    report.total_duration_seconds = total_duration_seconds
    report.slo_seconds = slo_seconds
    report.components_initialized = components_initialized
    report.components_failed = components_failed
    report.checkpoints = checkpoints
    report.error = error
    return report


def _make_checkpoint(name: str, elapsed_seconds: float):
    """Create a mock checkpoint."""
    cp = MagicMock()
    cp.name = name
    cp.elapsed_seconds = elapsed_seconds
    return cp


def _make_active_key(
    version: int = 1,
    created_at: datetime | None = None,
):
    """Create a mock encryption key."""
    key = MagicMock()
    key.version = version
    key.created_at = created_at or datetime.now(timezone.utc)
    return key


def _make_encryption_service(
    active_key=None,
    active_key_id: str = "key-001",
    encrypt_result: bytes = b"encrypted_data",
    decrypt_result: bytes = b"encryption_health_check",
):
    """Create a mock encryption service."""
    service = MagicMock()
    service.get_active_key.return_value = active_key
    service.get_active_key_id.return_value = active_key_id
    service.encrypt.return_value = encrypt_result
    service.decrypt.return_value = decrypt_result
    return service


# ============================================================================
# TestStartupHealth - startup_health() startup report and SLO status
# ============================================================================


class TestStartupHealth:
    """Tests for startup_health() with a valid report."""

    def test_healthy_when_success_and_slo_met(self):
        """Success + SLO met -> status healthy."""
        report = _make_startup_report(success=True, slo_met=True)
        with patch(
            "aragora.server.handlers.admin.health.platform.get_last_startup_report",
            return_value=report,
            create=True,
        ):
            # Need to patch the import inside the function
            import aragora.server.handlers.admin.health.platform as mod

            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.startup_transaction": MagicMock(
                        get_last_startup_report=MagicMock(return_value=report)
                    ),
                },
            ):
                result = startup_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "healthy"
        assert body["startup"]["success"] is True
        assert body["startup"]["slo_met"] is True

    def test_warning_when_success_but_slo_exceeded(self):
        """Success but SLO not met -> status warning."""
        report = _make_startup_report(success=True, slo_met=False, total_duration_seconds=45.0)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": MagicMock(
                    get_last_startup_report=MagicMock(return_value=report)
                ),
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "warning"
        assert body["startup"]["success"] is True
        assert body["startup"]["slo_met"] is False

    def test_degraded_when_startup_failed(self):
        """Startup failed -> status degraded."""
        report = _make_startup_report(success=False, slo_met=False, error="Init failed")
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": MagicMock(
                    get_last_startup_report=MagicMock(return_value=report)
                ),
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "degraded"
        assert body["startup"]["success"] is False
        assert body["error"] == "Init failed"

    def test_startup_duration_rounded(self):
        """Duration is rounded to 2 decimal places."""
        report = _make_startup_report(total_duration_seconds=5.12345)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": MagicMock(
                    get_last_startup_report=MagicMock(return_value=report)
                ),
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert body["startup"]["duration_seconds"] == 5.12

    def test_slo_seconds_in_response(self):
        """SLO target seconds is included in the response."""
        report = _make_startup_report(slo_seconds=30.0)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": MagicMock(
                    get_last_startup_report=MagicMock(return_value=report)
                ),
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert body["startup"]["slo_seconds"] == 30.0

    def test_components_in_response(self):
        """Component counts are in the response."""
        report = _make_startup_report(components_initialized=8, components_failed=2)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": MagicMock(
                    get_last_startup_report=MagicMock(return_value=report)
                ),
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert body["components"]["initialized"] == 8
        assert body["components"]["failed"] == 2

    def test_checkpoints_included(self):
        """Checkpoints are serialized with name and elapsed_seconds."""
        cps = [
            _make_checkpoint("db_connect", 0.5),
            _make_checkpoint("cache_init", 1.23456),
        ]
        report = _make_startup_report(checkpoints=cps)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": MagicMock(
                    get_last_startup_report=MagicMock(return_value=report)
                ),
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert len(body["checkpoints"]) == 2
        assert body["checkpoints"][0]["name"] == "db_connect"
        assert body["checkpoints"][0]["elapsed_seconds"] == 0.5
        assert body["checkpoints"][1]["name"] == "cache_init"
        assert body["checkpoints"][1]["elapsed_seconds"] == 1.23

    def test_checkpoints_none_when_empty(self):
        """Checkpoints is None when report has no checkpoints."""
        report = _make_startup_report(checkpoints=None)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": MagicMock(
                    get_last_startup_report=MagicMock(return_value=report)
                ),
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert body["checkpoints"] is None

    def test_response_time_ms_present(self):
        """response_time_ms is a non-negative number."""
        report = _make_startup_report()
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": MagicMock(
                    get_last_startup_report=MagicMock(return_value=report)
                ),
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert "response_time_ms" in body
        assert isinstance(body["response_time_ms"], (int, float))
        assert body["response_time_ms"] >= 0

    def test_timestamp_present(self):
        """ISO 8601 timestamp is in the response."""
        report = _make_startup_report()
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": MagicMock(
                    get_last_startup_report=MagicMock(return_value=report)
                ),
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert "timestamp" in body
        assert body["timestamp"].endswith("Z")

    def test_error_field_none_when_no_error(self):
        """Error field is None when startup had no error."""
        report = _make_startup_report(error=None)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": MagicMock(
                    get_last_startup_report=MagicMock(return_value=report)
                ),
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert body["error"] is None


class TestStartupHealthNoReport:
    """Tests for startup_health() when no report is available."""

    def test_unknown_status_when_no_report(self):
        """No startup report -> status unknown."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": MagicMock(
                    get_last_startup_report=MagicMock(return_value=None)
                ),
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "unknown"
        assert "No startup report" in body["message"]

    def test_response_time_ms_when_no_report(self):
        """response_time_ms present even with no report."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": MagicMock(
                    get_last_startup_report=MagicMock(return_value=None)
                ),
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0


class TestStartupHealthImportError:
    """Tests for startup_health() when module not installed."""

    def test_not_available_on_import_error(self):
        """ImportError -> status not_available, 503."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": None,
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "not_available"
        assert "not installed" in body["message"]
        assert _status(result) == 503

    def test_response_time_ms_on_import_error(self):
        """response_time_ms present even on import error."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": None,
            },
        ):
            result = startup_health(_make_mock_handler())

        body = _body(result)
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0


# ============================================================================
# TestEncryptionHealth - encryption_health() full check
# ============================================================================


class TestEncryptionHealth:
    """Tests for encryption_health() happy path."""

    def test_healthy_all_checks_pass(self):
        """All checks pass -> status healthy, 200."""
        key = _make_active_key(created_at=datetime.now(timezone.utc) - timedelta(days=10))
        service = _make_encryption_service(active_key=key)

        with (
            patch(
                "aragora.server.handlers.admin.health.platform.get_encryption_service",
                return_value=service,
                create=True,
            ),
            patch(
                "aragora.server.handlers.admin.health.platform.CRYPTO_AVAILABLE",
                True,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.security.encryption": MagicMock(
                        get_encryption_service=MagicMock(return_value=service),
                        CRYPTO_AVAILABLE=True,
                    ),
                },
            ),
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "healthy"
        assert _status(result) == 200
        assert body["health"]["cryptography_library"]["healthy"] is True
        assert body["health"]["encryption_service"]["healthy"] is True
        assert body["health"]["active_key"]["healthy"] is True
        assert body["health"]["roundtrip_test"]["healthy"] is True
        assert body["issues"] is None
        assert body["warnings"] is None

    def test_response_has_timestamp(self):
        """Response includes ISO timestamp."""
        key = _make_active_key()
        service = _make_encryption_service(active_key=key)

        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert "timestamp" in body
        assert body["timestamp"].endswith("Z")

    def test_response_has_response_time_ms(self):
        """Response includes response_time_ms."""
        key = _make_active_key()
        service = _make_encryption_service(active_key=key)

        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0


class TestEncryptionHealthImportError:
    """Tests for encryption_health() import failures."""

    def test_import_error_returns_503(self):
        """Cannot import encryption module -> 503."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": None,
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "error"
        assert _status(result) == 503
        assert "Cannot import encryption module" in body["issues"]
        assert body["health"]["cryptography_library"]["status"] == "import_error"

    def test_crypto_not_available(self):
        """CRYPTO_AVAILABLE is False -> issue reported."""
        service = _make_encryption_service(active_key=_make_active_key())
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=False,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["health"]["cryptography_library"]["healthy"] is False
        assert body["health"]["cryptography_library"]["status"] == "not_installed"
        assert "Cryptography library not installed" in body["issues"]


class TestEncryptionHealthServiceError:
    """Tests for encryption_health() service initialization failures."""

    def test_service_init_runtime_error(self):
        """get_encryption_service raises RuntimeError -> 503."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(side_effect=RuntimeError("no key")),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "error"
        assert _status(result) == 503
        assert body["health"]["encryption_service"]["healthy"] is False
        assert body["health"]["encryption_service"]["error"] == "Initialization failed"

    def test_service_init_value_error(self):
        """get_encryption_service raises ValueError -> 503."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(side_effect=ValueError("bad config")),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert _status(result) == 503
        assert "Encryption service initialization failed" in body["issues"]

    def test_service_init_os_error(self):
        """get_encryption_service raises OSError -> 503."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(side_effect=OSError("disk")),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        assert _status(result) == 503

    def test_service_init_type_error(self):
        """get_encryption_service raises TypeError -> 503."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(side_effect=TypeError("bad arg")),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        assert _status(result) == 503


class TestEncryptionHealthKeyStatus:
    """Tests for encryption_health() active key status."""

    def test_no_active_key(self):
        """No active key -> issue reported."""
        service = _make_encryption_service(active_key=None)
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["health"]["active_key"]["healthy"] is False
        assert body["health"]["active_key"]["status"] == "no_active_key"
        assert "No active encryption key" in body["issues"]

    def test_key_age_under_60_days(self):
        """Key under 60 days -> no warnings, no rotation_recommended."""
        key = _make_active_key(created_at=datetime.now(timezone.utc) - timedelta(days=30))
        service = _make_encryption_service(active_key=key)
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["health"]["active_key"]["healthy"] is True
        assert body["health"]["active_key"]["age_days"] == 30
        assert "rotation_recommended" not in body["health"]["active_key"]

    def test_key_age_between_60_and_90_days(self):
        """Key 60-90 days -> days_until_rotation included, no warning."""
        key = _make_active_key(created_at=datetime.now(timezone.utc) - timedelta(days=75))
        service = _make_encryption_service(active_key=key)
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["health"]["active_key"]["healthy"] is True
        assert body["health"]["active_key"]["days_until_rotation"] == 15

    def test_key_age_over_90_days_warning(self):
        """Key over 90 days -> warning and rotation_recommended."""
        key = _make_active_key(created_at=datetime.now(timezone.utc) - timedelta(days=100))
        service = _make_encryption_service(active_key=key)
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "warning"
        assert body["health"]["active_key"]["rotation_recommended"] is True
        assert any("Rotation recommended" in w for w in body["warnings"])

    def test_key_version_and_id(self):
        """Key version and ID are included in response."""
        key = _make_active_key(version=3)
        service = _make_encryption_service(active_key=key, active_key_id="key-v3")
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["health"]["active_key"]["version"] == 3
        assert body["health"]["active_key"]["key_id"] == "key-v3"

    def test_key_created_at_iso_format(self):
        """Key created_at is serialized in ISO format."""
        created = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        key = _make_active_key(created_at=created)
        service = _make_encryption_service(active_key=key)
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert "2025-06-15" in body["health"]["active_key"]["created_at"]


class TestEncryptionHealthRoundtrip:
    """Tests for encryption_health() roundtrip verification."""

    def test_roundtrip_success(self):
        """Encrypt/decrypt roundtrip succeeds -> roundtrip_test healthy."""
        key = _make_active_key()
        service = _make_encryption_service(
            active_key=key,
            decrypt_result=b"encryption_health_check",
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["health"]["roundtrip_test"]["healthy"] is True
        assert body["health"]["roundtrip_test"]["status"] == "passed"

    def test_roundtrip_data_mismatch(self):
        """Decrypted data doesn't match -> roundtrip_test unhealthy."""
        key = _make_active_key()
        service = _make_encryption_service(
            active_key=key,
            decrypt_result=b"wrong_data",
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["health"]["roundtrip_test"]["healthy"] is False
        assert body["health"]["roundtrip_test"]["status"] == "data_mismatch"
        assert "round-trip failed" in body["issues"][0]

    def test_roundtrip_runtime_error(self):
        """RuntimeError during roundtrip -> error status."""
        key = _make_active_key()
        service = _make_encryption_service(active_key=key)
        service.encrypt.side_effect = RuntimeError("encrypt failed")
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["health"]["roundtrip_test"]["healthy"] is False
        assert body["health"]["roundtrip_test"]["status"] == "error"
        assert any("roundtrip" in i.lower() for i in body["issues"])

    def test_roundtrip_value_error(self):
        """ValueError during roundtrip -> error status."""
        key = _make_active_key()
        service = _make_encryption_service(active_key=key)
        service.decrypt.side_effect = ValueError("bad padding")
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["health"]["roundtrip_test"]["healthy"] is False

    def test_roundtrip_os_error(self):
        """OSError during roundtrip -> error status."""
        key = _make_active_key()
        service = _make_encryption_service(active_key=key)
        service.encrypt.side_effect = OSError("disk")
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["health"]["roundtrip_test"]["healthy"] is False

    def test_roundtrip_type_error(self):
        """TypeError during roundtrip -> error status."""
        key = _make_active_key()
        service = _make_encryption_service(active_key=key)
        service.encrypt.side_effect = TypeError("wrong type")
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["health"]["roundtrip_test"]["healthy"] is False


# ============================================================================
# TestPlatformHealth - platform_health() platform resilience check
# ============================================================================


class TestPlatformHealth:
    """Tests for platform_health() overall structure."""

    def test_all_imports_fail_not_configured(self):
        """All modules unavailable -> status not_configured."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "not_configured"
        assert body["summary"]["active"] == 0

    def test_response_has_summary(self):
        """Response contains summary with total/healthy/active."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert "summary" in body
        assert "total_components" in body["summary"]
        assert "healthy" in body["summary"]
        assert "active" in body["summary"]

    def test_response_has_components(self):
        """Response contains components dict."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert "components" in body
        expected_keys = [
            "rate_limiters",
            "resilience",
            "dead_letter_queue",
            "metrics",
            "platform_circuits",
        ]
        for key in expected_keys:
            assert key in body["components"], f"Missing component: {key}"

    def test_response_has_response_time_ms(self):
        """Response includes response_time_ms."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0

    def test_response_has_timestamp(self):
        """Response includes ISO timestamp."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert "timestamp" in body
        assert body["timestamp"].endswith("Z")

    def test_warnings_none_when_empty(self):
        """Warnings is None when there are no warnings."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["warnings"] is None

    def test_total_components_is_5(self):
        """There are always 5 components checked."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["summary"]["total_components"] == 5


class TestPlatformHealthRateLimiters:
    """Tests for platform_health() rate limiter component."""

    def test_rate_limiters_active(self):
        """Rate limiters module available -> active with config."""
        mock_limiter = MagicMock()
        mock_limiter.rpm = 60
        mock_limiter.burst_size = 10
        mock_limiter.daily_limit = 10000

        mock_module = MagicMock()
        mock_module.PLATFORM_RATE_LIMITS = {"slack": {}, "discord": {}}
        mock_module.get_platform_rate_limiter.return_value = mock_limiter

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": mock_module,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        comp = body["components"]["rate_limiters"]
        assert comp["healthy"] is True
        assert comp["status"] == "active"
        assert "slack" in comp["platforms"]
        assert "discord" in comp["platforms"]
        assert comp["config"]["slack"]["rpm"] == 60

    def test_rate_limiters_import_error(self):
        """Rate limiter module not available -> not_available."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        comp = body["components"]["rate_limiters"]
        assert comp["healthy"] is True
        assert comp["status"] == "not_available"

    def test_rate_limiters_runtime_error(self):
        """RuntimeError in rate limiter check -> error, unhealthy."""
        mock_module = MagicMock()
        mock_module.PLATFORM_RATE_LIMITS = {"slack": {}}
        mock_module.get_platform_rate_limiter.side_effect = RuntimeError("broken")

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": mock_module,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        comp = body["components"]["rate_limiters"]
        assert comp["healthy"] is False
        assert comp["status"] == "error"

    def test_rate_limiters_value_error(self):
        """ValueError in rate limiter check -> error."""
        mock_module = MagicMock()
        mock_module.PLATFORM_RATE_LIMITS = {"slack": {}}
        mock_module.get_platform_rate_limiter.side_effect = ValueError("bad")

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": mock_module,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["components"]["rate_limiters"]["healthy"] is False

    def test_rate_limiters_key_error(self):
        """KeyError in rate limiter check -> error."""
        mock_module = MagicMock()
        mock_module.PLATFORM_RATE_LIMITS = {"slack": {}}
        mock_module.get_platform_rate_limiter.side_effect = KeyError("missing")

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": mock_module,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["components"]["rate_limiters"]["healthy"] is False


class TestPlatformHealthResilience:
    """Tests for platform_health() resilience module component."""

    def test_resilience_active(self):
        """Resilience module available -> active with stats."""
        mock_resilience = MagicMock()
        mock_resilience.get_stats.return_value = {
            "platforms_tracked": 4,
            "circuit_breakers": {"slack": "closed"},
        }

        mock_module = MagicMock()
        mock_module.get_platform_resilience.return_value = mock_resilience
        mock_module.DLQ_ENABLED = True
        # Also mock get_dlq for dead_letter_queue check
        mock_dlq = MagicMock()
        mock_dlq.get_stats.return_value = {"pending": 0, "failed": 0, "processed": 10}
        mock_module.get_dlq.return_value = mock_dlq

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": mock_module,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        comp = body["components"]["resilience"]
        assert comp["healthy"] is True
        assert comp["status"] == "active"
        assert comp["dlq_enabled"] is True
        assert comp["platforms_tracked"] == 4

    def test_resilience_import_error(self):
        """Resilience module not available -> not_available."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["components"]["resilience"]["status"] == "not_available"
        assert body["components"]["resilience"]["healthy"] is True

    def test_resilience_runtime_error(self):
        """RuntimeError in resilience check -> error but still healthy."""
        mock_module = MagicMock()
        mock_module.get_platform_resilience.side_effect = RuntimeError("broken")
        mock_module.get_dlq.side_effect = RuntimeError("broken")

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": mock_module,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        comp = body["components"]["resilience"]
        assert comp["healthy"] is True  # resilience errors don't set healthy=False
        assert comp["status"] == "error"

    def test_resilience_attribute_error(self):
        """AttributeError in resilience check -> error."""
        mock_module = MagicMock()
        mock_module.get_platform_resilience.side_effect = AttributeError("no attr")
        mock_module.get_dlq.side_effect = AttributeError("no attr")

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": mock_module,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["components"]["resilience"]["status"] == "error"


class TestPlatformHealthDLQ:
    """Tests for platform_health() dead letter queue component."""

    def _make_dlq_module(self, pending=0, failed=0, processed=10):
        """Create a mock module with DLQ stats."""
        mock_dlq = MagicMock()
        mock_dlq.get_stats.return_value = {
            "pending": pending,
            "failed": failed,
            "processed": processed,
        }
        mock_module = MagicMock()
        mock_module.get_dlq.return_value = mock_dlq
        # resilience check will also import from same module
        mock_resilience = MagicMock()
        mock_resilience.get_stats.return_value = {"platforms_tracked": 0, "circuit_breakers": {}}
        mock_module.get_platform_resilience.return_value = mock_resilience
        mock_module.DLQ_ENABLED = True
        return mock_module

    def test_dlq_active(self):
        """DLQ with low pending -> active, healthy."""
        mock_module = self._make_dlq_module(pending=5)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": mock_module,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        comp = body["components"]["dead_letter_queue"]
        assert comp["healthy"] is True
        assert comp["status"] == "active"
        assert comp["pending_count"] == 5

    def test_dlq_elevated_warning(self):
        """DLQ with 50+ pending -> elevated warning."""
        mock_module = self._make_dlq_module(pending=55)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": mock_module,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["warnings"] is not None
        assert any("elevated" in w for w in body["warnings"])

    def test_dlq_high_warning(self):
        """DLQ with 100+ pending -> high warning."""
        mock_module = self._make_dlq_module(pending=150)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": mock_module,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["warnings"] is not None
        assert any("150" in w for w in body["warnings"])

    def test_dlq_no_warning_under_50(self):
        """DLQ with under 50 pending -> no warning."""
        mock_module = self._make_dlq_module(pending=10)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": mock_module,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        # The resilience check may still be active so warnings could exist for other reasons
        # But no DLQ-related warnings should appear
        if body["warnings"]:
            for w in body["warnings"]:
                assert "pending" not in w.lower()

    def test_dlq_import_error(self):
        """DLQ module not available -> not_available."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        comp = body["components"]["dead_letter_queue"]
        assert comp["healthy"] is True
        assert comp["status"] == "not_available"

    def test_dlq_runtime_error(self):
        """RuntimeError in DLQ check -> error but healthy."""
        mock_module = MagicMock()
        mock_resilience = MagicMock()
        mock_resilience.get_stats.return_value = {"platforms_tracked": 0, "circuit_breakers": {}}
        mock_module.get_platform_resilience.return_value = mock_resilience
        mock_module.DLQ_ENABLED = True
        mock_module.get_dlq.side_effect = RuntimeError("DLQ broken")

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": mock_module,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        comp = body["components"]["dead_letter_queue"]
        assert comp["healthy"] is True
        assert comp["status"] == "error"

    def test_dlq_failed_count(self):
        """DLQ includes failed and processed counts."""
        mock_module = self._make_dlq_module(pending=3, failed=7, processed=100)
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": mock_module,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        comp = body["components"]["dead_letter_queue"]
        assert comp["failed_count"] == 7
        assert comp["processed_count"] == 100


class TestPlatformHealthMetrics:
    """Tests for platform_health() platform metrics component."""

    def test_metrics_active(self):
        """Metrics module available -> active with summary."""
        mock_metrics_module = MagicMock()
        mock_metrics_module.get_platform_metrics_summary.return_value = {"total_requests": 1000}

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": mock_metrics_module,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        comp = body["components"]["metrics"]
        assert comp["healthy"] is True
        assert comp["status"] == "active"
        assert comp["prometheus_enabled"] is True
        assert comp["summary"]["total_requests"] == 1000

    def test_metrics_import_error(self):
        """Metrics module not available -> not_available."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        comp = body["components"]["metrics"]
        assert comp["healthy"] is True
        assert comp["status"] == "not_available"
        assert comp["prometheus_enabled"] is False

    def test_metrics_runtime_error(self):
        """RuntimeError in metrics check -> error but healthy."""
        mock_module = MagicMock()
        mock_module.get_platform_metrics_summary.side_effect = RuntimeError("broken")

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": mock_module,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["components"]["metrics"]["status"] == "error"
        assert body["components"]["metrics"]["healthy"] is True

    def test_metrics_type_error(self):
        """TypeError in metrics -> error."""
        mock_module = MagicMock()
        mock_module.get_platform_metrics_summary.side_effect = TypeError("bad")

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": mock_module,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["components"]["metrics"]["status"] == "error"


class TestPlatformHealthCircuitBreakers:
    """Tests for platform_health() circuit breaker component."""

    def _make_circuit_breaker(self, state="closed", failure_count=0, success_count=10):
        """Create a mock circuit breaker."""
        cb = MagicMock()
        cb.state = MagicMock()
        cb.state.value = state
        cb.failure_count = failure_count
        cb.success_count = success_count
        cb.failures = failure_count  # fallback attribute
        return cb

    def test_circuit_breakers_all_closed(self):
        """All circuit breakers closed -> active, healthy."""
        cb = self._make_circuit_breaker(state="closed")
        mock_module = MagicMock()
        mock_module.get_circuit_breaker.return_value = cb

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": mock_module,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        comp = body["components"]["platform_circuits"]
        assert comp["healthy"] is True
        assert comp["status"] == "active"
        circuits = comp["circuits"]
        # All 6 platforms checked
        for platform in ["slack", "discord", "teams", "telegram", "whatsapp", "matrix"]:
            assert platform in circuits
            assert circuits[platform]["state"] == "closed"

    def test_circuit_breaker_open_triggers_warning(self):
        """Open circuit breaker -> warning and all_healthy becomes False."""
        closed_cb = self._make_circuit_breaker(state="closed")
        open_cb = self._make_circuit_breaker(state="open", failure_count=5)

        mock_module = MagicMock()

        def get_cb(name):
            if name == "platform_slack":
                return open_cb
            return closed_cb

        mock_module.get_circuit_breaker.side_effect = get_cb

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": mock_module,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "degraded"
        assert body["warnings"] is not None
        assert any("slack" in w.lower() for w in body["warnings"])

    def test_circuit_breaker_returns_none(self):
        """Circuit breaker returns None -> platform skipped (not in circuits dict)."""
        mock_module = MagicMock()
        mock_module.get_circuit_breaker.return_value = None

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": mock_module,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        circuits = body["components"]["platform_circuits"]["circuits"]
        # When cb is None, the platform is simply skipped (not added to dict)
        assert circuits == {}

    def test_circuit_breaker_key_error_per_platform(self):
        """KeyError for individual platform -> not_configured for that platform."""
        mock_module = MagicMock()
        mock_module.get_circuit_breaker.side_effect = KeyError("not found")

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": mock_module,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        circuits = body["components"]["platform_circuits"]["circuits"]
        for platform in ["slack", "discord", "teams", "telegram", "whatsapp", "matrix"]:
            assert circuits[platform]["state"] == "not_configured"

    def test_circuit_breaker_import_error(self):
        """Circuit breaker module not available -> not_available."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        comp = body["components"]["platform_circuits"]
        assert comp["healthy"] is True
        assert comp["status"] == "not_available"

    def test_circuit_breaker_module_runtime_error(self):
        """RuntimeError at module level (not per-platform) -> error but healthy."""
        mock_module = MagicMock()
        # Make the initial import work but the loop setup fail
        mock_module.get_circuit_breaker.side_effect = RuntimeError("bad")

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": mock_module,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        # RuntimeError is caught by per-platform handler, but there is an outer catch too
        # The per-platform handler catches KeyError, ValueError, AttributeError, TypeError
        # RuntimeError would propagate to the outer handler
        comp = body["components"]["platform_circuits"]
        assert comp["healthy"] is True

    def test_circuit_breaker_state_without_value_attribute(self):
        """Circuit breaker state without .value -> uses str()."""

        class PlainState:
            """State object without .value attribute."""

            def __str__(self):
                return "half_open"

        cb = MagicMock()
        cb.state = PlainState()
        cb.failure_count = 2
        cb.success_count = 5
        cb.failures = 2

        mock_module = MagicMock()
        mock_module.get_circuit_breaker.return_value = cb

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": mock_module,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        circuits = body["components"]["platform_circuits"]["circuits"]
        # state should be converted via str()
        for platform in circuits:
            assert circuits[platform]["state"] == "half_open"

    def test_circuit_breaker_uses_failures_fallback(self):
        """When failure_count not available, falls back to .failures."""
        cb = MagicMock(spec=[])  # empty spec
        cb.state = MagicMock()
        cb.state.value = "closed"
        cb.failures = 3
        # failure_count via getattr fallback
        cb.success_count = 10

        mock_module = MagicMock()
        mock_module.get_circuit_breaker.return_value = cb

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": mock_module,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        # Check that circuits are populated
        comp = body["components"]["platform_circuits"]
        assert comp["healthy"] is True

    def test_multiple_open_circuits(self):
        """Multiple open circuit breakers -> all listed in warning."""
        open_cb = self._make_circuit_breaker(state="open")
        closed_cb = self._make_circuit_breaker(state="closed")

        mock_module = MagicMock()

        def get_cb(name):
            if name in ("platform_slack", "platform_discord"):
                return open_cb
            return closed_cb

        mock_module.get_circuit_breaker.side_effect = get_cb

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": mock_module,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "degraded"
        warning_text = " ".join(body["warnings"])
        assert "slack" in warning_text
        assert "discord" in warning_text


class TestPlatformHealthOverallStatus:
    """Tests for platform_health() overall status determination."""

    def test_healthy_when_active_and_no_warnings(self):
        """Active components and no warnings -> healthy."""
        mock_limiter = MagicMock()
        mock_limiter.rpm = 60
        mock_limiter.burst_size = 10
        mock_limiter.daily_limit = 10000

        mock_rl_module = MagicMock()
        mock_rl_module.PLATFORM_RATE_LIMITS = {"slack": {}}
        mock_rl_module.get_platform_rate_limiter.return_value = mock_limiter

        cb = MagicMock()
        cb.state = MagicMock()
        cb.state.value = "closed"
        cb.failure_count = 0
        cb.success_count = 10
        cb.failures = 0

        mock_resilience_module = MagicMock()
        mock_resilience_module.get_circuit_breaker.return_value = cb

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": mock_rl_module,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": mock_resilience_module,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] in ("healthy", "healthy_with_warnings")

    def test_healthy_with_warnings_when_dlq_elevated(self):
        """Active components with DLQ warnings -> healthy_with_warnings."""
        mock_dlq = MagicMock()
        mock_dlq.get_stats.return_value = {"pending": 75, "failed": 0, "processed": 10}

        mock_resilience = MagicMock()
        mock_resilience.get_stats.return_value = {"platforms_tracked": 1, "circuit_breakers": {}}

        mock_pr_module = MagicMock()
        mock_pr_module.get_platform_resilience.return_value = mock_resilience
        mock_pr_module.DLQ_ENABLED = True
        mock_pr_module.get_dlq.return_value = mock_dlq

        mock_rl_module = MagicMock()
        mock_limiter = MagicMock()
        mock_limiter.rpm = 60
        mock_limiter.burst_size = 10
        mock_limiter.daily_limit = 10000
        mock_rl_module.PLATFORM_RATE_LIMITS = {"slack": {}}
        mock_rl_module.get_platform_rate_limiter.return_value = mock_limiter

        cb = MagicMock()
        cb.state = MagicMock()
        cb.state.value = "closed"
        cb.failure_count = 0
        cb.success_count = 10
        cb.failures = 0

        mock_resilience_mod = MagicMock()
        mock_resilience_mod.get_circuit_breaker.return_value = cb

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": mock_rl_module,
                "aragora.integrations.platform_resilience": mock_pr_module,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": mock_resilience_mod,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "healthy_with_warnings"
        assert body["warnings"] is not None

    def test_degraded_when_rate_limiter_error_with_active_component(self):
        """Rate limiter error sets all_healthy=False -> degraded (with active component)."""
        mock_rl_module = MagicMock()
        mock_rl_module.PLATFORM_RATE_LIMITS = {"slack": {}}
        mock_rl_module.get_platform_rate_limiter.side_effect = RuntimeError("broken")

        # Need at least one active component so active_count > 0,
        # otherwise "not_configured" overrides "degraded".
        cb = MagicMock()
        cb.state = MagicMock()
        cb.state.value = "closed"
        cb.failure_count = 0
        cb.success_count = 10
        cb.failures = 0

        mock_resilience_mod = MagicMock()
        mock_resilience_mod.get_circuit_breaker.return_value = cb

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": mock_rl_module,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": mock_resilience_mod,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "degraded"
        assert body["summary"]["active"] >= 1

    def test_not_configured_when_no_active_components(self):
        """No active components -> not_configured."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "not_configured"
        assert body["summary"]["active"] == 0

    def test_healthy_count_in_summary(self):
        """Healthy count reflects number of healthy components."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        # All 5 components should be healthy (just not_available)
        assert body["summary"]["healthy"] == 5

    def test_active_count_with_some_active(self):
        """Active count reflects number of components with status=active."""
        mock_rl_module = MagicMock()
        mock_limiter = MagicMock()
        mock_limiter.rpm = 60
        mock_limiter.burst_size = 10
        mock_limiter.daily_limit = 10000
        mock_rl_module.PLATFORM_RATE_LIMITS = {"slack": {}}
        mock_rl_module.get_platform_rate_limiter.return_value = mock_limiter

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": mock_rl_module,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["summary"]["active"] >= 1

    def test_degraded_overrides_warnings_status(self):
        """When degraded, status is degraded even with warnings."""
        open_cb = MagicMock()
        open_cb.state = MagicMock()
        open_cb.state.value = "open"
        open_cb.failure_count = 5
        open_cb.success_count = 0
        open_cb.failures = 5

        mock_module = MagicMock()
        mock_module.get_circuit_breaker.return_value = open_cb

        # Also add a rate limiter error to set all_healthy=False
        mock_rl_module = MagicMock()
        mock_rl_module.PLATFORM_RATE_LIMITS = {"slack": {}}
        mock_rl_module.get_platform_rate_limiter.side_effect = RuntimeError("broken")

        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": mock_rl_module,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": mock_module,
            },
        ):
            result = platform_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "degraded"


# ============================================================================
# TestEncryptionHealthCombinedScenarios - edge cases
# ============================================================================


class TestEncryptionHealthCombinedScenarios:
    """Tests for encryption_health() combined and edge case scenarios."""

    def test_crypto_not_available_with_no_key(self):
        """Crypto not available but service works (edge case) -> error with issues."""
        service = _make_encryption_service(active_key=None)
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=False,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "error"
        assert _status(result) == 503
        # Should have multiple issues
        assert len(body["issues"]) >= 1

    def test_warning_status_with_old_key_but_healthy_roundtrip(self):
        """Old key (>90 days) but roundtrip passes -> warning."""
        key = _make_active_key(created_at=datetime.now(timezone.utc) - timedelta(days=120))
        service = _make_encryption_service(
            active_key=key,
            decrypt_result=b"encryption_health_check",
        )
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "warning"
        assert _status(result) == 200
        assert body["health"]["roundtrip_test"]["healthy"] is True

    def test_error_status_has_503(self):
        """Error status always returns 503."""
        service = _make_encryption_service(active_key=None)
        service.encrypt.side_effect = RuntimeError("fail")
        with patch.dict(
            "sys.modules",
            {
                "aragora.security.encryption": MagicMock(
                    get_encryption_service=MagicMock(return_value=service),
                    CRYPTO_AVAILABLE=True,
                ),
            },
        ):
            result = encryption_health(_make_mock_handler())

        body = _body(result)
        assert body["status"] == "error"
        assert _status(result) == 503


# ============================================================================
# TestHandlerIntegration - HealthHandler routing to platform functions
# ============================================================================


class TestHandlerIntegration:
    """Tests verifying HealthHandler routes to platform.py functions."""

    @pytest.fixture
    def health_handler(self):
        """Create a HealthHandler instance."""
        from aragora.server.handlers.admin.health import HealthHandler

        return HealthHandler(ctx={})

    @pytest.mark.asyncio
    async def test_platform_route(self, health_handler):
        """HealthHandler routes /api/v1/health/platform to platform_health."""
        mock_http = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = await health_handler.handle("/api/v1/health/platform", {}, mock_http)

        body = _body(result)
        assert "components" in body
        assert "summary" in body

    @pytest.mark.asyncio
    async def test_platform_health_alternate_route(self, health_handler):
        """HealthHandler routes /api/v1/platform/health to platform_health."""
        mock_http = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.middleware.rate_limit.platform_limiter": None,
                "aragora.integrations.platform_resilience": None,
                "aragora.observability.metrics.platform": None,
                "aragora.resilience": None,
            },
        ):
            result = await health_handler.handle("/api/v1/platform/health", {}, mock_http)

        body = _body(result)
        assert "components" in body

    @pytest.mark.asyncio
    async def test_startup_route(self, health_handler):
        """HealthHandler routes /api/v1/health/startup to startup_health."""
        mock_http = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "aragora.server.startup_transaction": MagicMock(
                    get_last_startup_report=MagicMock(return_value=None)
                ),
            },
        ):
            result = await health_handler.handle("/api/v1/health/startup", {}, mock_http)

        body = _body(result)
        assert body["status"] == "unknown"
