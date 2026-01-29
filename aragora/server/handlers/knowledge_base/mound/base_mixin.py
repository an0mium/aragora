"""
Base protocol for Knowledge Mound handler operations.

Provides the common interface that all mound operation mixins expect
from the concrete handler implementation. Uses Protocol (PEP 544) for
structural typing instead of abstract base classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

@runtime_checkable
class KnowledgeMoundHandlerProtocol(Protocol):
    """
    Protocol defining the interface that Knowledge Mound operation mixins expect.

    The concrete handler (KnowledgeMoundHandler) must implement these methods.
    Mixins use this protocol for type hints instead of defining stub methods.

    This uses structural typing - any class implementing _get_mound() and
    having a ctx attribute will satisfy this protocol.

    Attributes:
        ctx: Server context dictionary
    """

    ctx: dict[str, Any]

    def _get_mound(self) -> Optional["KnowledgeMound"]:
        """
        Get the Knowledge Mound instance.

        Returns:
            KnowledgeMound instance or None if not available
        """
        ...

# Backward compatibility alias
KnowledgeMoundMixinBase = KnowledgeMoundHandlerProtocol

__all__ = ["KnowledgeMoundHandlerProtocol", "KnowledgeMoundMixinBase"]
