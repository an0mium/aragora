"""
Base mixin for Knowledge Mound handler operations.

Provides the common interface that all mound operation mixins must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound


class KnowledgeMoundMixinBase(ABC):
    """
    Abstract base class for Knowledge Mound operation mixins.

    All mound operation mixins should inherit from this class to ensure
    they implement the required interface for accessing the mound instance.

    Attributes:
        ctx: Server context dictionary
    """

    ctx: Dict[str, Any]

    @abstractmethod
    def _get_mound(self) -> Optional["KnowledgeMound"]:
        """
        Get the Knowledge Mound instance.

        Returns:
            KnowledgeMound instance or None if not available

        Raises:
            NotImplementedError: If not implemented by concrete handler
        """
        raise NotImplementedError("Subclass must implement _get_mound")


__all__ = ["KnowledgeMoundMixinBase"]
