"""
Canonical spec containers for bead and convoy creation.

These lightweight dataclasses avoid importing core models at runtime to
prevent circular imports. They are intended for CLI and adapter layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.nomic.beads import BeadPriority, BeadType
    from aragora.nomic.convoys import ConvoyPriority


@dataclass
class BeadSpec:
    """Input spec for creating a bead."""

    title: str
    description: str = ""
    priority: "BeadPriority | int | None" = None
    bead_type: "BeadType | str | None" = None
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvoySpec:
    """Input spec for creating a convoy from bead specs."""

    title: str
    description: str = ""
    beads: list[BeadSpec] = field(default_factory=list)
    priority: "ConvoyPriority | int | None" = None
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
