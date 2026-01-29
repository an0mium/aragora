"""
Canonical Bead Store.

Re-exports the authoritative BeadStore implementation from aragora.nomic.beads.
All consumers should import from this module or from aragora.nomic.stores.

The Nomic BeadStore is the single source of truth for bead persistence.
Workspace and Gastown layers use adapters that delegate to this store.
"""

from aragora.nomic.beads import (
    Bead,
    BeadPriority,
    BeadStatus,
    BeadStore,
    BeadType,
)

__all__ = [
    "Bead",
    "BeadPriority",
    "BeadStatus",
    "BeadStore",
    "BeadType",
]
