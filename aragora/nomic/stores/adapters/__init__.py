"""
Store adapters for backward compatibility.

The workspace and gastown layers each define their own data models and status
enums. These adapters map between layer-specific models and the canonical
Nomic store models.

Adapter chain:
    Gastown ConvoyTracker -> Workspace ConvoyTracker -> Nomic ConvoyManager
    Gastown BeadManager   -> Workspace BeadManager   -> Nomic BeadStore
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.workspace.bead import BeadManager as WorkspaceBeadAdapter
    from aragora.workspace.convoy import ConvoyTracker as WorkspaceConvoyAdapter

__all__ = [
    "WorkspaceBeadAdapter",
    "WorkspaceConvoyAdapter",
]
