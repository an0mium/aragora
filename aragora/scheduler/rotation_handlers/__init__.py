"""
Secrets Rotation Handlers

Provider-specific handlers for automated secret rotation.
Each handler implements the rotation lifecycle:
1. Generate new credentials (if supported by provider)
2. Validate new credentials work
3. Update secret storage
4. Revoke old credentials (with grace period)
"""

from .base import RotationError, RotationHandler, RotationResult, RotationStatus
from .database import DatabaseRotationHandler

__all__ = [
    "DatabaseRotationHandler",
    "RotationError",
    "RotationHandler",
    "RotationResult",
    "RotationStatus",
]
