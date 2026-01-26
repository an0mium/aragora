"""
Protocol definitions for Knowledge Mound mixin classes.

These protocols define the expected interface that host classes must implement
when using Knowledge Mound operation mixins like FederationMixin, SyncMixin, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


@runtime_checkable
class KnowledgeMoundHostProtocol(Protocol):
    """
    Protocol defining the expected interface for Knowledge Mound host classes.

    Mixin classes like FederationMixin, SyncMixin, and SharingMixin expect
    the host class to provide these attributes and methods.
    """

    # Core attributes
    workspace_id: str
    config: Any  # MoundConfig
    _initialized: bool

    # Memory subsystems
    _continuum: Any  # ContinuumMemory
    _consensus: Any  # ConsensusMemory
    _facts: Any  # FactsStore or similar
    _evidence: Any  # EvidenceStore
    _critique: Any  # CritiqueStore

    def _ensure_initialized(self) -> None:
        """Ensure the mound is initialized before operations."""
        ...

    async def query(
        self,
        query: str,
        **kwargs: Any,
    ) -> Any:
        """Query the knowledge mound."""
        ...

    async def store(self, item: Any) -> Any:
        """Store an item in the knowledge mound."""
        ...
