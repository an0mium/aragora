"""
Canonical Primitive Stores.

This package is the single source of truth for Aragora's work-tracking
primitives: Beads (atomic work units) and Convoys (work batches).

Import from here rather than reaching into aragora.nomic.beads or
aragora.nomic.convoys directly:

    from aragora.nomic.stores import BeadStore, ConvoyStore, Bead, Convoy

Layer hierarchy:
    aragora.nomic.stores          <- Canonical (this package)
    aragora.workspace.bead/convoy <- Workspace adapters (status mapping, metadata)
    aragora.extensions.gastown    <- Gastown wrappers (artifacts, handoffs)
"""

from .bead_store import (
    Bead,
    BeadPriority,
    BeadStatus,
    BeadStore,
    BeadType,
)
from .convoy_store import (
    Convoy,
    ConvoyManager,
    ConvoyPriority,
    ConvoyProgress,
    ConvoyStatus,
    ConvoyStore,
)
from .specs import BeadSpec, ConvoySpec
from .protocols import BeadRecord, ConvoyRecord
from aragora.nomic.beads import create_bead_store, get_bead_store, reset_bead_store
from aragora.nomic.convoys import get_convoy_manager, reset_convoy_manager

__all__ = [
    # Bead primitives
    "Bead",
    "BeadPriority",
    "BeadStatus",
    "BeadStore",
    "BeadType",
    "BeadSpec",
    # Convoy primitives
    "Convoy",
    "ConvoyManager",
    "ConvoyPriority",
    "ConvoyProgress",
    "ConvoyStatus",
    "ConvoyStore",
    "ConvoySpec",
    # Cross-layer protocols
    "BeadRecord",
    "ConvoyRecord",
    # Store helpers
    "create_bead_store",
    "get_bead_store",
    "reset_bead_store",
    "get_convoy_manager",
    "reset_convoy_manager",
]
