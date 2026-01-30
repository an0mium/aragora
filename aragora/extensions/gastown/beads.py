"""
Gastown Bead Adapter — re-exports from workspace and nomic bead layers.

Provides the gastown extension entry point for bead management so
dashboard handlers import from the extension rather than reaching
into nomic internals directly.

Two BeadManager variants are available:
- ``BeadManager`` — workspace-level adapter (JSONL-backed, status mapping)
- ``NomicBeadManager`` — nomic-level manager (used by dashboard for
  status iteration and priority filtering)
"""

from __future__ import annotations

from aragora.workspace.bead import (
    Bead,
    BeadManager,
    BeadStatus,
    generate_bead_id,
)

# Re-export nomic-level classes used by dashboard for queue stats
from aragora.nomic.stores import (
    BeadPriority,
    BeadStatus as NomicBeadStatus,
    BeadStore as NomicBeadManager,
)

__all__ = [
    "Bead",
    "BeadManager",
    "BeadStatus",
    "BeadPriority",
    "NomicBeadManager",
    "NomicBeadStatus",
    "generate_bead_id",
]
