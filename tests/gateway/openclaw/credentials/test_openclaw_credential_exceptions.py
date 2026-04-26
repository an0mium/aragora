"""Unit tests for OpenClaw credential exception classes."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_exceptions_module():
    module_path = (
        Path(__file__).resolve().parents[4]
        / "aragora"
        / "gateway"
        / "openclaw"
        / "credentials"
        / "exceptions.py"
    )
    spec = spec_from_file_location(
        "openclaw_credentials_exceptions_under_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exceptions = _load_exceptions_module()

CredentialVaultError = exceptions.CredentialVaultError
CredentialNotFoundError = exceptions.CredentialNotFoundError
CredentialAccessDeniedError = exceptions.CredentialAccessDeniedError
CredentialExpiredError = exceptions.CredentialExpiredError
CredentialRateLimitedError = exceptions.CredentialRateLimitedError
TenantIsolationError = exceptions.TenantIsolationError
EncryptionError = exceptions.EncryptionError


def test_module_exports_expected_exception_names() -> None:
    expected = {
        "CredentialVaultError",
        "CredentialNotFoundError",
        "CredentialAccessDeniedError",
        "CredentialExpiredError",
        "CredentialRateLimitedError",
        "TenantIsolationError",
        "EncryptionError",
    }

    assert set(exceptions.__all__) == expected


def test_credential_vault_error_uses_empty_default_message() -> None:
    error = CredentialVaultError()

    assert error.message == ""
    assert str(error) == ""
    assert error.args == ("",)


def test_credential_vault_error_preserves_explicit_message() -> None:
    error = CredentialVaultError("vault failure")

    assert error.message == "vault failure"
    assert str(error) == "vault failure"
    assert error.args == ("vault failure",)


def test_credential_not_found_error_inherits_base_error() -> None:
    error = CredentialNotFoundError("missing credential")

    assert isinstance(error, CredentialVaultError)
    assert error.message == "missing credential"
    assert str(error) == "missing credential"


def test_credential_access_denied_error_uses_default_context_values() -> None:
    error = CredentialAccessDeniedError("denied")

    assert isinstance(error, CredentialVaultError)
    assert error.credential_id is None
    assert error.user_id is None
    assert error.reason == "permission_denied"
    assert str(error) == "denied"


def test_credential_access_denied_error_preserves_custom_context() -> None:
    error = CredentialAccessDeniedError(
        "restricted",
        credential_id="cred-123",
        user_id="user-456",
        reason="role_required",
    )

    assert error.credential_id == "cred-123"
    assert error.user_id == "user-456"
    assert error.reason == "role_required"
    assert error.message == "restricted"


def test_credential_expired_error_inherits_base_error() -> None:
    error = CredentialExpiredError("credential expired")

    assert isinstance(error, CredentialVaultError)
    assert error.message == "credential expired"
    assert str(error) == "credential expired"


def test_credential_rate_limited_error_uses_default_retry_after_seconds() -> None:
    error = CredentialRateLimitedError("slow down")

    assert isinstance(error, CredentialVaultError)
    assert error.retry_after_seconds == 60
    assert error.message == "slow down"


def test_credential_rate_limited_error_preserves_custom_retry_after_seconds() -> None:
    error = CredentialRateLimitedError("back off", retry_after_seconds=15)

    assert error.retry_after_seconds == 15
    assert str(error) == "back off"


def test_tenant_isolation_error_inherits_base_error() -> None:
    error = TenantIsolationError("cross-tenant access")

    assert isinstance(error, CredentialVaultError)
    assert error.message == "cross-tenant access"
    assert str(error) == "cross-tenant access"


def test_encryption_error_inherits_base_error() -> None:
    error = EncryptionError("decryption failed")

    assert isinstance(error, CredentialVaultError)
    assert error.message == "decryption failed"
    assert str(error) == "decryption failed"
