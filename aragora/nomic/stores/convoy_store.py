"""
Canonical Convoy Store.

Re-exports the authoritative ConvoyManager implementation from aragora.nomic.convoys.
All consumers should import from this module or from aragora.nomic.stores.

The Nomic ConvoyManager is the single source of truth for convoy persistence.
Workspace and Gastown layers use adapters that delegate to this store.

Note: The canonical class is named ConvoyManager (not ConvoyStore) for historical
reasons. The alias ConvoyStore is provided for consistency with BeadStore naming.
"""

from aragora.nomic.convoys import (
    Convoy,
    ConvoyManager,
    ConvoyPriority,
    ConvoyProgress,
    ConvoyStatus,
)

# Alias for naming consistency with BeadStore
ConvoyStore = ConvoyManager

__all__ = [
    "Convoy",
    "ConvoyManager",
    "ConvoyPriority",
    "ConvoyProgress",
    "ConvoyStatus",
    "ConvoyStore",
]
