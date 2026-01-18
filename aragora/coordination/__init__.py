"""
Cross-Workspace Coordination Module.

Enables workflows and agent operations that span multiple workspaces
with proper isolation, permission management, and federation.

Features:
- Cross-workspace data sharing with consent
- Federated agent execution
- Multi-workspace workflow orchestration
- Secure inter-workspace communication
- Permission delegation and scoping
"""

from aragora.coordination.cross_workspace import (
    CrossWorkspaceCoordinator,
    FederatedWorkspace,
    FederationPolicy,
    CrossWorkspaceRequest,
    CrossWorkspaceResult,
    DataSharingConsent,
    SharingScope,
)

__all__ = [
    "CrossWorkspaceCoordinator",
    "FederatedWorkspace",
    "FederationPolicy",
    "CrossWorkspaceRequest",
    "CrossWorkspaceResult",
    "DataSharingConsent",
    "SharingScope",
]
