"""
Workspace Handler Package - Enterprise Privacy and Data Isolation APIs.

This package provides API endpoints for workspace and privacy management:
- Workspace creation and management
- Data isolation and access control
- Retention policy management
- Sensitivity classification
- Privacy audit logging

The package is structured as follows:
- workspace_utils.py: Circuit breaker and validation utilities
- (future) handler.py: Main WorkspaceHandler class

For backwards compatibility, all public exports are available directly from this package.

Stability: STABLE

Example usage:
    from aragora.server.handlers.workspace import WorkspaceHandler
    from aragora.server.handlers.workspace import WorkspaceCircuitBreaker
    from aragora.server.handlers.workspace import get_workspace_circuit_breaker_status
"""

from __future__ import annotations

# Import WorkspaceHandler and feature flags from the workspace_module (sibling of this package)
# Also import symbols that tests need to patch for backwards compatibility
from aragora.server.handlers.workspace_module import (
    WorkspaceHandler,
    RBAC_AVAILABLE,
    PROFILES_AVAILABLE,
    # Re-export for test patching compatibility
    extract_user_from_request,
    PrivacyAuditLog,
    SensitivityLevel,
    DataIsolationManager,
    RetentionPolicyManager,
    SensitivityClassifier,
)

# Import utilities from workspace_utils submodule
from .workspace_utils import (
    WorkspaceCircuitBreaker,
    _get_workspace_circuit_breaker,
    get_workspace_circuit_breaker_status,
    _validate_workspace_id,
    _validate_policy_id,
    _validate_user_id,
)

# Import sensitivity classification handlers
from .sensitivity import (
    handle_classify_content,
    handle_get_level_policy,
)

# Import audit log handlers
from .audit import (
    handle_query_audit,
    handle_audit_report,
    handle_verify_integrity,
    handle_actor_history,
    handle_resource_history,
    handle_denied_access,
)

__all__ = [
    # Main handler
    "WorkspaceHandler",
    # Feature flags (for test patching compatibility)
    "RBAC_AVAILABLE",
    "PROFILES_AVAILABLE",
    # Re-exported for test patching compatibility
    "extract_user_from_request",
    "PrivacyAuditLog",
    "SensitivityLevel",
    "DataIsolationManager",
    "RetentionPolicyManager",
    "SensitivityClassifier",
    # Circuit breaker
    "WorkspaceCircuitBreaker",
    "get_workspace_circuit_breaker_status",
    "_get_workspace_circuit_breaker",
    # Validation utilities
    "_validate_workspace_id",
    "_validate_policy_id",
    "_validate_user_id",
    # Sensitivity classification handlers
    "handle_classify_content",
    "handle_get_level_policy",
    # Audit log handlers
    "handle_query_audit",
    "handle_audit_report",
    "handle_verify_integrity",
    "handle_actor_history",
    "handle_resource_history",
    "handle_denied_access",
]
