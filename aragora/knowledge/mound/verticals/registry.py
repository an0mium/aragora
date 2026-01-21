"""
Registry for vertical knowledge modules.

Provides factory methods for creating and managing vertical-specific
knowledge modules within the Knowledge Mound.
"""

from __future__ import annotations

import logging
from typing import Any, Type

from aragora.knowledge.mound.verticals.base import BaseVerticalKnowledge

logger = logging.getLogger(__name__)


class VerticalRegistry:
    """
    Registry for vertical knowledge modules.

    Maintains a registry of vertical implementations and provides
    factory methods for creating instances.

    Usage:
        # Registration via decorator
        @VerticalRegistry.register
        class SoftwareKnowledge(BaseVerticalKnowledge):
            ...

        # Get a vertical module
        software = VerticalRegistry.get("software")

        # List available verticals
        verticals = VerticalRegistry.list_all()
    """

    _registry: dict[str, Type[BaseVerticalKnowledge]] = {}
    _instances: dict[str, BaseVerticalKnowledge] = {}

    @classmethod
    def register(
        cls,
        vertical_class: Type[BaseVerticalKnowledge],
    ) -> Type[BaseVerticalKnowledge]:
        """
        Register a vertical knowledge module class.

        Can be used as a decorator:
            @VerticalRegistry.register
            class SoftwareKnowledge(BaseVerticalKnowledge):
                ...

        Args:
            vertical_class: Class implementing BaseVerticalKnowledge

        Returns:
            The registered class (for decorator use)
        """
        # Instantiate to get vertical_id
        instance = vertical_class()
        vertical_id = instance.vertical_id

        cls._registry[vertical_id] = vertical_class
        logger.debug(f"Registered vertical: {vertical_id} -> {vertical_class.__name__}")

        return vertical_class

    @classmethod
    def register_instance(cls, instance: BaseVerticalKnowledge) -> None:
        """
        Register a pre-configured instance.

        Args:
            instance: Configured vertical instance
        """
        vertical_id = instance.vertical_id
        cls._instances[vertical_id] = instance
        cls._registry[vertical_id] = type(instance)
        logger.debug(f"Registered vertical instance: {vertical_id}")

    @classmethod
    def unregister(cls, vertical_id: str) -> bool:
        """
        Unregister a vertical.

        Args:
            vertical_id: ID to unregister

        Returns:
            True if was registered
        """
        was_registered = vertical_id in cls._registry
        cls._registry.pop(vertical_id, None)
        cls._instances.pop(vertical_id, None)
        return was_registered

    @classmethod
    def get(cls, vertical_id: str) -> BaseVerticalKnowledge | None:
        """
        Get a vertical module instance.

        Returns cached instance if available, otherwise creates new one.

        Args:
            vertical_id: Vertical identifier

        Returns:
            Vertical instance or None if not registered
        """
        # Check for cached instance
        if vertical_id in cls._instances:
            return cls._instances[vertical_id]

        # Create new instance
        vertical_class = cls._registry.get(vertical_id)
        if vertical_class:
            instance = vertical_class()
            cls._instances[vertical_id] = instance
            return instance

        return None

    @classmethod
    def get_or_raise(cls, vertical_id: str) -> BaseVerticalKnowledge:
        """
        Get a vertical module instance, raising if not found.

        Args:
            vertical_id: Vertical identifier

        Returns:
            Vertical instance

        Raises:
            KeyError: If vertical not registered
        """
        instance = cls.get(vertical_id)
        if instance is None:
            available = list(cls._registry.keys())
            raise KeyError(f"Unknown vertical: '{vertical_id}'. Available: {available}")
        return instance

    @classmethod
    def list_all(cls) -> list[str]:
        """List all registered vertical IDs."""
        return list(cls._registry.keys())

    @classmethod
    def get_all(cls) -> dict[str, BaseVerticalKnowledge]:
        """Get all vertical instances."""
        result = {}
        for vertical_id in cls._registry:
            instance = cls.get(vertical_id)
            if instance:
                result[vertical_id] = instance
        return result

    @classmethod
    def is_registered(cls, vertical_id: str) -> bool:
        """Check if a vertical is registered."""
        return vertical_id in cls._registry

    @classmethod
    def get_capabilities(cls, vertical_id: str) -> dict[str, Any] | None:
        """Get capabilities for a vertical."""
        instance = cls.get(vertical_id)
        if instance:
            caps = instance.capabilities
            return {
                "pattern_detection": caps.supports_pattern_detection,
                "cross_reference": caps.supports_cross_reference,
                "compliance_check": caps.supports_compliance_check,
                "requires_llm": caps.requires_llm,
                "pattern_categories": caps.pattern_categories,
                "compliance_frameworks": caps.compliance_frameworks,
                "document_types": caps.document_types,
            }
        return None

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._registry.clear()
        cls._instances.clear()


def _register_default_verticals() -> None:
    """Register default vertical implementations."""
    try:
        from aragora.knowledge.mound.verticals.software import SoftwareKnowledge

        VerticalRegistry.register(SoftwareKnowledge)
    except ImportError:
        logger.debug("Software vertical not available")

    try:
        from aragora.knowledge.mound.verticals.legal import LegalKnowledge

        VerticalRegistry.register(LegalKnowledge)
    except ImportError:
        logger.debug("Legal vertical not available")

    try:
        from aragora.knowledge.mound.verticals.healthcare import HealthcareKnowledge

        VerticalRegistry.register(HealthcareKnowledge)
    except ImportError:
        logger.debug("Healthcare vertical not available")

    try:
        from aragora.knowledge.mound.verticals.accounting import AccountingKnowledge

        VerticalRegistry.register(AccountingKnowledge)
    except ImportError:
        logger.debug("Accounting vertical not available")

    try:
        from aragora.knowledge.mound.verticals.research import ResearchKnowledge

        VerticalRegistry.register(ResearchKnowledge)
    except ImportError:
        logger.debug("Research vertical not available")


# Register defaults on module load
_register_default_verticals()
