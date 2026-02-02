"""Backward-compatible sensitivity exports for workspace handlers."""

from __future__ import annotations

from aragora.privacy import (
    DataIsolationManager,
    PrivacyAuditLog,
    RetentionPolicyManager,
    SensitivityClassifier,
    SensitivityLevel,
)

__all__ = [
    "DataIsolationManager",
    "PrivacyAuditLog",
    "RetentionPolicyManager",
    "SensitivityClassifier",
    "SensitivityLevel",
]
