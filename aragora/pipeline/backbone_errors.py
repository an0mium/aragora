"""Backbone persistence error helpers for fail-closed interactive flows."""

from __future__ import annotations

FAIL_CLOSED_BACKBONE_MESSAGE = "Backbone persistence failed; interactive execution blocked"


class BackbonePersistenceError(RuntimeError):
    """Raised when interactive execution cannot guarantee backbone persistence."""


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
    "ensure_backbone_persisted",
]
