"""
Structural protocols for convoy and bead data models.

These protocols define the minimal field contracts that all convoy/bead
implementations must satisfy, regardless of layer. They enable type-safe
cross-layer interoperability without requiring inheritance.

The property-based design allows each layer to satisfy the contract via
simple property accessors that alias their existing fields, without
requiring any field renames.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConvoyRecord(Protocol):
    """Minimal structural contract for any convoy data object.

    Every layer's Convoy dataclass should satisfy this protocol by
    implementing the required properties. Field name normalization
    is handled by each implementation's property accessors.

    Properties use the `convoy_` prefix to avoid collisions with
    existing dataclass fields that may have the same base name.
    """

    @property
    def convoy_id(self) -> str:
        """Unique identifier for the convoy."""
        ...

    @property
    def convoy_title(self) -> str:
        """Human-readable title/name for the convoy."""
        ...

    @property
    def convoy_description(self) -> str:
        """Description of the convoy's purpose."""
        ...

    @property
    def convoy_bead_ids(self) -> list[str]:
        """List of bead IDs in this convoy."""
        ...

    @property
    def convoy_status_value(self) -> str:
        """String value of the convoy's status enum."""
        ...

    @property
    def convoy_created_at(self) -> datetime:
        """When the convoy was created (as datetime)."""
        ...

    @property
    def convoy_updated_at(self) -> datetime:
        """When the convoy was last updated (as datetime)."""
        ...

    @property
    def convoy_assigned_agents(self) -> list[str]:
        """List of agent IDs assigned to this convoy."""
        ...

    @property
    def convoy_error(self) -> str | None:
        """Error message if convoy failed, else None."""
        ...

    @property
    def convoy_metadata(self) -> dict[str, Any]:
        """Arbitrary metadata dictionary."""
        ...


@runtime_checkable
class BeadRecord(Protocol):
    """Minimal structural contract for any bead data object.

    Similar to ConvoyRecord, this allows cross-layer interoperability
    for bead objects without requiring a shared base class.
    """

    @property
    def bead_id(self) -> str:
        """Unique identifier for the bead."""
        ...

    @property
    def bead_convoy_id(self) -> str | None:
        """ID of the convoy this bead belongs to, if any."""
        ...

    @property
    def bead_status_value(self) -> str:
        """String value of the bead's status enum."""
        ...

    @property
    def bead_content(self) -> str:
        """The bead's content/description."""
        ...

    @property
    def bead_created_at(self) -> datetime:
        """When the bead was created (as datetime)."""
        ...

    @property
    def bead_metadata(self) -> dict[str, Any]:
        """Arbitrary metadata dictionary."""
        ...
