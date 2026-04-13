"""
OpenClaw Credential Vault - Exception classes.
"""

from __future__ import annotations

__all__ = [
    "CredentialVaultError",
    "CredentialNotFoundError",
    "CredentialAccessDeniedError",
    "CredentialExpiredError",
    "CredentialRateLimitedError",
    "TenantIsolationError",
    "EncryptionError",
]


class CredentialVaultError(Exception):
    """Base exception for credential vault errors."""

    message: str

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message


class CredentialNotFoundError(CredentialVaultError):
    """Credential not found in vault."""

    pass


class CredentialAccessDeniedError(CredentialVaultError):
    """Access to credential denied."""

    credential_id: str | None
    user_id: str | None
    reason: str

    def __init__(
        self,
        message: str,
        credential_id: str | None = None,
        user_id: str | None = None,
        reason: str = "permission_denied",
    ):
        super().__init__(message)
        self.credential_id = credential_id
        self.user_id = user_id
        self.reason = reason


class CredentialExpiredError(CredentialVaultError):
    """Credential has expired."""

    pass


class CredentialRateLimitedError(CredentialVaultError):
    """Too many credential access attempts."""

    retry_after_seconds: int

    def __init__(self, message: str, retry_after_seconds: int = 60):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class TenantIsolationError(CredentialVaultError):
    """Cross-tenant credential access attempted."""

    pass


class EncryptionError(CredentialVaultError):
    """Encryption or decryption failed."""

    pass
