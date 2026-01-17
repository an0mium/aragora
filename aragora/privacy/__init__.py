"""
Privacy and Data Isolation Module.

Provides enterprise-grade privacy controls:
- Workspace data isolation
- Sensitivity classification
- Data retention policies
- Audit-grade access logging

Usage:
    from aragora.privacy import DataIsolationManager, SensitivityClassifier

    isolation = DataIsolationManager()
    workspace = await isolation.create_isolated_workspace("org_123")

    classifier = SensitivityClassifier()
    level = await classifier.classify(document)
"""

from aragora.privacy.isolation import (
    DataIsolationManager,
    IsolationConfig,
    Workspace,
    WorkspacePermission,
    AccessDeniedException,
)
from aragora.privacy.retention import (
    RetentionPolicyManager,
    RetentionPolicy,
    RetentionAction,
    DeletionReport,
    RetentionViolation,
)
from aragora.privacy.classifier import (
    SensitivityClassifier,
    SensitivityLevel,
    ClassificationResult,
    ClassificationConfig,
)
from aragora.privacy.audit_log import (
    PrivacyAuditLog,
    AuditEntry,
    AuditAction,
    AuditOutcome,
)

__all__ = [
    # Isolation
    "DataIsolationManager",
    "IsolationConfig",
    "Workspace",
    "WorkspacePermission",
    "AccessDeniedException",
    # Retention
    "RetentionPolicyManager",
    "RetentionPolicy",
    "RetentionAction",
    "DeletionReport",
    "RetentionViolation",
    # Classification
    "SensitivityClassifier",
    "SensitivityLevel",
    "ClassificationResult",
    "ClassificationConfig",
    # Audit
    "PrivacyAuditLog",
    "AuditEntry",
    "AuditAction",
    "AuditOutcome",
]
