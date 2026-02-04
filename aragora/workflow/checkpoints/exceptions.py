"""
Checkpoint store exceptions.
"""

from __future__ import annotations


class CheckpointValidationError(Exception):
    """Raised when checkpoint validation fails."""

    pass


class ConnectionTimeoutError(Exception):
    """Raised when database connection times out."""

    pass
