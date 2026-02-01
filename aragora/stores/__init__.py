"""
Canonical store interfaces and helpers.

This package defines the canonical store access points for core
orchestration primitives (convoys/beads/workspaces) and the
gateway/inbox runtime. Use these helpers to avoid bypassing the
canonical persistence layer.
"""

from aragora.stores.canonical import (
    CanonicalGatewayStores,
    CanonicalWorkspaceStores,
    get_canonical_gateway_stores,
    get_canonical_workspace_stores,
)

__all__ = [
    "CanonicalGatewayStores",
    "CanonicalWorkspaceStores",
    "get_canonical_gateway_stores",
    "get_canonical_workspace_stores",
]
