"""Backbone persistence error helpers for fail-closed interactive flows."""

from __future__ import annotations

FAIL_CLOSED_BACKBONE_MESSAGE = "Backbone persistence failed; interactive execution blocked"

FAIL_CLOSED_HTTP_STATUS = 503


class BackbonePersistenceError(RuntimeError):
    """Raised when interactive execution cannot guarantee backbone persistence.

    Carries a ``status_code`` so HTTP handlers can return an explicit
    503 Service Unavailable without hard-coding the value.
    """

    status_code: int = FAIL_CLOSED_HTTP_STATUS

    def __init__(self, message: str | None = None, *, status_code: int | None = None) -> None:
        super().__init__(message or FAIL_CLOSED_BACKBONE_MESSAGE)
        if status_code is not None:
            self.status_code = status_code


class FailClosedBackboneError(BackbonePersistenceError):
    """Raised when interactive handlers must fail closed on backbone persistence errors."""


def ensure_backbone_persisted(
    ok: bool,
    message: str = FAIL_CLOSED_BACKBONE_MESSAGE,
) -> None:
    """Raise a typed error when a required backbone write did not persist."""
    if not ok:
        raise FailClosedBackboneError(message)


__all__ = [
    "BackbonePersistenceError",
    "FailClosedBackboneError",
    "FAIL_CLOSED_BACKBONE_MESSAGE",
    "FAIL_CLOSED_HTTP_STATUS",
    "ensure_backbone_persisted",
]
