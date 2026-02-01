"""Registry for external agent framework adapters.

Provides decorator-based registration and factory pattern for creating
adapter instances. Follows the same patterns as Aragora's agent registry
and KM adapter factory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from .base import ExternalAgentAdapter
    from .config import ExternalAgentConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="ExternalAgentAdapter")


@dataclass(frozen=True)
class ExternalAdapterSpec:
    """Specification for a registered external agent adapter.

    Contains metadata about the adapter for discovery, documentation,
    and factory creation.
    """

    name: str
    adapter_class: type["ExternalAgentAdapter"]
    config_class: type["ExternalAgentConfig"]
    description: str | None = None
    requires: str | None = None  # External dependency description
    env_vars: str | None = None  # Required environment variables


class ExternalAgentRegistry:
    """Registry for external agent adapters.

    Provides decorator-based registration and factory pattern for
    creating adapter instances.

    Usage:
        # Registration (done in adapter modules)
        @ExternalAgentRegistry.register(
            "openhands",
            config_class=OpenHandsConfig,
            description="OpenHands autonomous coding agent",
            requires="OpenHands server running",
        )
        class OpenHandsAdapter(ExternalAgentAdapter):
            ...

        # Creation
        adapter = ExternalAgentRegistry.create(
            "openhands",
            config=OpenHandsConfig(base_url="http://localhost:3000")
        )

        # Listing
        available = ExternalAgentRegistry.list_all()
    """

    _registry: dict[str, ExternalAdapterSpec] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        config_class: type["ExternalAgentConfig"],
        description: str | None = None,
        requires: str | None = None,
        env_vars: str | None = None,
    ) -> Callable[[type[T]], type[T]]:
        """Decorator to register an external agent adapter.

        Args:
            name: Unique adapter identifier (e.g., "openhands").
            config_class: Configuration dataclass for this adapter.
            description: Human-readable description.
            requires: External dependency description.
            env_vars: Required environment variables.

        Returns:
            Decorator function.

        Example:
            @ExternalAgentRegistry.register(
                "openhands",
                config_class=OpenHandsConfig,
                description="OpenHands AI coding agent",
            )
            class OpenHandsAdapter(ExternalAgentAdapter):
                adapter_name = "openhands"
                ...
        """

        def decorator(adapter_cls: type[T]) -> type[T]:
            spec = ExternalAdapterSpec(
                name=name,
                adapter_class=adapter_cls,
                config_class=config_class,
                description=description,
                requires=requires,
                env_vars=env_vars,
            )
            cls._registry[name] = spec
            logger.info(f"Registered external agent adapter: {name}")
            return adapter_cls

        return decorator

    @classmethod
    def create(
        cls,
        adapter_name: str,
        config: "ExternalAgentConfig | None" = None,
        **kwargs: Any,
    ) -> "ExternalAgentAdapter":
        """Create an adapter instance by name.

        Args:
            adapter_name: Registered adapter name.
            config: Optional configuration (uses defaults if not provided).
            **kwargs: Additional arguments passed to adapter constructor.

        Returns:
            Adapter instance.

        Raises:
            ValueError: If adapter_name is not registered.

        Example:
            adapter = ExternalAgentRegistry.create("openhands")
            task_id = await adapter.submit_task(request)
        """
        if adapter_name not in cls._registry:
            valid = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown adapter: {adapter_name}. Valid adapters: {valid}")

        spec = cls._registry[adapter_name]

        # Use provided config or create default
        if config is None:
            config = spec.config_class()

        return spec.adapter_class(config=config, **kwargs)

    @classmethod
    def is_registered(cls, adapter_name: str) -> bool:
        """Check if an adapter is registered.

        Args:
            adapter_name: Name to check.

        Returns:
            True if registered.
        """
        return adapter_name in cls._registry

    @classmethod
    def get_spec(cls, adapter_name: str) -> ExternalAdapterSpec | None:
        """Get the spec for a registered adapter.

        Args:
            adapter_name: Registered adapter name.

        Returns:
            AdapterSpec or None if not found.
        """
        return cls._registry.get(adapter_name)

    @classmethod
    def list_all(cls) -> dict[str, dict[str, Any]]:
        """List all registered adapters with metadata.

        Returns:
            Dictionary mapping adapter names to their metadata.
        """
        return {
            name: {
                "description": spec.description,
                "requires": spec.requires,
                "env_vars": spec.env_vars,
                "config_class": spec.config_class.__name__,
            }
            for name, spec in cls._registry.items()
        }

    @classmethod
    def list_specs(cls) -> list["ExternalAdapterSpec"]:
        """List all registered adapter specs.

        Returns:
            List of ExternalAdapterSpec objects.
        """
        return list(cls._registry.values())

    @classmethod
    def get_registered_names(cls) -> list[str]:
        """Get list of registered adapter names.

        Returns:
            Sorted list of adapter names.
        """
        return sorted(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Useful for testing."""
        cls._registry.clear()
        logger.debug("Cleared all external agent adapter registrations")


def register_all_adapters() -> None:
    """Import all adapter modules to trigger registration.

    Called at startup to ensure all adapters are registered.
    Safe to call multiple times (registrations are idempotent).
    """
    # OpenHands adapter
    try:
        from aragora.agents.external.adapters import openhands  # noqa: F401

        logger.debug("Loaded OpenHands adapter")
    except ImportError as e:
        logger.debug(f"OpenHands adapter not available: {e}")

    # AutoGPT adapter (future)
    try:
        from aragora.agents.external.adapters import autogpt  # noqa: F401

        logger.debug("Loaded AutoGPT adapter")
    except ImportError as e:
        logger.debug(f"AutoGPT adapter not available: {e}")

    # CrewAI adapter (future)
    try:
        from aragora.agents.external.adapters import crewai  # noqa: F401

        logger.debug("Loaded CrewAI adapter")
    except ImportError as e:
        logger.debug(f"CrewAI adapter not available: {e}")
