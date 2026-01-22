"""Custom exceptions for the Aragora SDK."""

from __future__ import annotations

from typing import Any


class AragoraError(Exception):
    """Base exception for all Aragora SDK errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.details = details or {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({self.message!r}, code={self.code!r}, status={self.status})"
        )


class AragoraConnectionError(AragoraError):
    """Raised when unable to connect to the Aragora server."""

    def __init__(self, message: str = "Failed to connect to Aragora server") -> None:
        super().__init__(message, code="CONNECTION_ERROR")


class AragoraAuthenticationError(AragoraError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, code="AUTHENTICATION_ERROR", status=401)


class AragoraNotFoundError(AragoraError):
    """Raised when a resource is not found."""

    def __init__(self, resource: str, resource_id: str) -> None:
        message = f"{resource} not found: {resource_id}"
        super().__init__(message, code="NOT_FOUND", status=404)
        self.resource = resource
        self.resource_id = resource_id


class AragoraValidationError(AragoraError):
    """Raised when request validation fails."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="VALIDATION_ERROR", status=400, details=details)


class AragoraTimeoutError(AragoraError):
    """Raised when a request times out."""

    def __init__(self, message: str = "Request timed out") -> None:
        super().__init__(message, code="TIMEOUT")
