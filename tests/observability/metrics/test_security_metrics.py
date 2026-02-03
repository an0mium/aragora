"""Tests for observability/metrics/security.py — security metrics."""

from unittest.mock import patch

import pytest

from aragora.observability.metrics import security as mod
from aragora.observability.metrics.security import (
    init_security_metrics,
    record_auth_attempt,
    record_auth_failure,
    record_blocked_request,
    record_encrypted_field,
    record_encryption_error,
    record_encryption_operation,
    record_key_operation,
    record_key_rotation,
    record_migration_duration,
    record_migration_error,
    record_migration_record,
    record_rbac_cache_hit,
    record_rbac_cache_miss,
    record_rbac_decision,
    record_rbac_denial,
    record_rbac_evaluation_latency,
    record_secret_access,
    record_secret_decryption,
    record_security_alert,
    record_security_incident,
    record_sensitive_field_operation,
    set_active_keys,
    set_active_sessions,
    track_auth_attempt,
    track_encryption_operation,
    track_migration,
    track_rbac_evaluation,
)


@pytest.fixture(autouse=True)
def _reset_module():
    mod._initialized = False
    yield
    mod._initialized = False


@pytest.fixture()
def _init_noop():
    with patch("aragora.observability.metrics.security.get_metrics_config") as mock_cfg:
        mock_cfg.return_value.enabled = False
        init_security_metrics()


class TestInitialization:
    def test_init_disabled(self, _init_noop):
        assert mod._initialized is True
        assert mod.ENCRYPTION_OPERATIONS_TOTAL is not None

    def test_init_idempotent(self, _init_noop):
        first = mod.ENCRYPTION_OPERATIONS_TOTAL
        init_security_metrics()
        assert mod.ENCRYPTION_OPERATIONS_TOTAL is first

    def test_init_returns_false_when_disabled(self):
        with patch("aragora.observability.metrics.security.get_metrics_config") as mock_cfg:
            mock_cfg.return_value.enabled = False
            result = init_security_metrics()
            assert result is False


class TestEncryptionMetrics:
    def test_record_encryption_operation(self, _init_noop):
        record_encryption_operation("encrypt", success=True, latency_seconds=0.01)

    def test_record_encryption_operation_failure(self, _init_noop):
        record_encryption_operation("decrypt", success=False, latency_seconds=0.05)

    def test_record_encryption_error(self, _init_noop):
        record_encryption_error("decrypt", "invalid_key")

    def test_record_encrypted_field(self, _init_noop):
        record_encrypted_field("api_key", "integrations")

    def test_track_encryption_operation_success(self, _init_noop):
        with track_encryption_operation("encrypt"):
            pass

    def test_track_encryption_operation_failure(self, _init_noop):
        with pytest.raises(ValueError):
            with track_encryption_operation("decrypt"):
                raise ValueError("bad data")


class TestKeyManagementMetrics:
    def test_record_key_operation(self, _init_noop):
        record_key_operation("generate", success=True)
        record_key_operation("load", success=False)

    def test_record_key_rotation(self, _init_noop):
        record_key_rotation("key-1", success=True, latency_seconds=2.5)

    def test_set_active_keys(self, _init_noop):
        set_active_keys(master=2, session=10, ephemeral=50)


class TestAuthenticationMetrics:
    def test_record_auth_attempt_success(self, _init_noop):
        record_auth_attempt("jwt", success=True, latency_seconds=0.01)

    def test_record_auth_attempt_no_latency(self, _init_noop):
        record_auth_attempt("api_key", success=False)

    def test_record_auth_failure(self, _init_noop):
        record_auth_failure("jwt", "expired_token")

    def test_set_active_sessions(self, _init_noop):
        set_active_sessions(jwt=100, api_key=20, oauth=5)

    def test_track_auth_attempt_success(self, _init_noop):
        with track_auth_attempt("jwt"):
            pass

    def test_track_auth_attempt_failure(self, _init_noop):
        with pytest.raises(RuntimeError):
            with track_auth_attempt("oauth"):
                raise RuntimeError("auth failed")


class TestRBACMetrics:
    def test_record_rbac_decision_granted(self, _init_noop):
        record_rbac_decision("debates.create", granted=True)

    def test_record_rbac_decision_denied(self, _init_noop):
        record_rbac_decision("admin.delete", granted=False)

    def test_record_rbac_denial(self, _init_noop):
        record_rbac_denial("admin.delete", "viewer")

    def test_record_rbac_evaluation_latency(self, _init_noop):
        record_rbac_evaluation_latency(0.001)

    def test_track_rbac_evaluation(self, _init_noop):
        with track_rbac_evaluation():
            pass

    def test_record_rbac_cache_hit(self, _init_noop):
        record_rbac_cache_hit("permission", "l1")

    def test_record_rbac_cache_miss(self, _init_noop):
        record_rbac_cache_miss("role")


class TestSecretAccessMetrics:
    def test_record_secret_access(self, _init_noop):
        record_secret_access("api_key", "read")

    def test_record_secret_decryption(self, _init_noop):
        record_secret_decryption("integrations", "api_key")

    def test_record_sensitive_field_operation(self, _init_noop):
        record_sensitive_field_operation("password", "encrypt")


class TestSecurityIncidentMetrics:
    def test_record_security_incident(self, _init_noop):
        record_security_incident("high", "brute_force")

    def test_record_security_alert(self, _init_noop):
        record_security_alert("anomaly", "slack")

    def test_record_blocked_request(self, _init_noop):
        record_blocked_request("rate_limit", "192.168.1.1")


class TestMigrationMetrics:
    def test_record_migration_record(self, _init_noop):
        record_migration_record("integrations", success=True)
        record_migration_record("webhooks", success=False)

    def test_record_migration_error(self, _init_noop):
        record_migration_error("integrations", "decryption_failed")

    def test_record_migration_duration(self, _init_noop):
        record_migration_duration("integrations", 15.5)

    def test_track_migration(self, _init_noop):
        with track_migration("webhooks"):
            pass
