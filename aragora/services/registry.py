"""
Centralized Service Registry for dependency management.

This module provides a thread-safe service registry that replaces scattered
global singletons throughout the codebase. Benefits:
- Single location for all services
- Easy to mock services in tests via reset()
- Type-safe service resolution
- Lazy initialization support via factories

Usage:
    # Direct registration
    registry = ServiceRegistry.get()
    registry.register(TierManager, tier_manager_instance)

    # Factory registration (lazy init)
    registry.register_factory(TierManager, lambda: TierManager())

    # Get service
    tier_manager = registry.resolve(TierManager)

    # Module-level convenience functions
    from aragora.services import get_service, register_service
    register_service(TierManager, my_instance)
    tier_manager = get_service(TierManager)
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional, Type, TypeVar, overload

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceNotFoundError(Exception):
    """Raised when a requested service is not registered."""

    def __init__(self, service_type: Type) -> None:
        self.service_type = service_type
        super().__init__(f"Service not found: {service_type.__name__}")


class ServiceRegistry:
    """
    Thread-safe singleton registry for application services.

    Provides centralized management of services that would otherwise be
    scattered global singletons. Supports both direct instance registration
    and lazy factory registration.

    Example:
        # Get the singleton registry
        registry = ServiceRegistry.get()

        # Register a service instance
        registry.register(TierManager, tier_manager)

        # Or register a factory for lazy initialization
        registry.register_factory(TierManager, TierManager)

        # Resolve a service
        tier_manager = registry.resolve(TierManager)

        # Check if service exists
        if registry.has(TierManager):
            ...

        # Reset for testing
        registry.reset()
    """

    _instance: Optional["ServiceRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the service registry."""
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[[], Any]] = {}
        self._service_lock = threading.RLock()

    @classmethod
    def get(cls) -> "ServiceRegistry":
        """
        Get the singleton registry instance.

        Thread-safe lazy initialization.

        Returns:
            The global ServiceRegistry instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.debug("ServiceRegistry initialized")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Reset the registry, clearing all services.

        Primarily used for testing to ensure clean state between tests.
        Also useful when reconfiguring the application.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._services.clear()
                cls._instance._factories.clear()
                logger.debug("ServiceRegistry reset - all services cleared")
            cls._instance = None

    def register(self, service_type: Type[T], instance: T) -> None:
        """
        Register a service instance.

        Args:
            service_type: The type to register the service under.
            instance: The service instance.

        Example:
            registry.register(TierManager, TierManager())
        """
        with self._service_lock:
            if service_type in self._services:
                logger.warning(
                    f"Overwriting existing service: {service_type.__name__}"
                )
            self._services[service_type] = instance
            # Remove factory if we're providing direct instance
            self._factories.pop(service_type, None)
            logger.debug(f"Registered service: {service_type.__name__}")

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[[], T],
    ) -> None:
        """
        Register a factory for lazy service initialization.

        The factory will be called on first resolve() and the result cached.

        Args:
            service_type: The type to register the factory under.
            factory: A callable that returns the service instance.

        Example:
            registry.register_factory(TierManager, TierManager)
            # or with configuration:
            registry.register_factory(
                TierManager,
                lambda: TierManager(custom_config)
            )
        """
        with self._service_lock:
            self._factories[service_type] = factory
            # Remove cached instance to trigger factory on next resolve
            self._services.pop(service_type, None)
            logger.debug(f"Registered factory: {service_type.__name__}")

    @overload
    def resolve(self, service_type: Type[T]) -> T: ...

    @overload
    def resolve(self, service_type: Type[T], default: T) -> T: ...

    @overload
    def resolve(self, service_type: Type[T], default: None) -> Optional[T]: ...

    def resolve(
        self,
        service_type: Type[T],
        default: Optional[T] = None,
    ) -> Optional[T]:
        """
        Resolve a service by type.

        If a factory is registered, it will be called and the result cached.

        Args:
            service_type: The type of service to resolve.
            default: Default value if service not found (None raises exception).

        Returns:
            The service instance.

        Raises:
            ServiceNotFoundError: If no service or factory is registered
                and no default is provided.

        Example:
            tier_manager = registry.resolve(TierManager)
            # With default:
            tier_manager = registry.resolve(TierManager, default=TierManager())
        """
        with self._service_lock:
            # Check for cached instance
            if service_type in self._services:
                return self._services[service_type]

            # Check for factory
            if service_type in self._factories:
                factory = self._factories[service_type]
                instance = factory()
                self._services[service_type] = instance
                logger.debug(
                    f"Lazily initialized service: {service_type.__name__}"
                )
                return instance

            # Not found - use default or raise
            if default is not None:
                return default

            raise ServiceNotFoundError(service_type)

    def has(self, service_type: Type) -> bool:
        """
        Check if a service is registered.

        Returns True if either an instance or factory is registered.

        Args:
            service_type: The type to check.

        Returns:
            True if service is available.
        """
        with self._service_lock:
            return (
                service_type in self._services
                or service_type in self._factories
            )

    def unregister(self, service_type: Type) -> bool:
        """
        Unregister a service.

        Args:
            service_type: The type to unregister.

        Returns:
            True if service was found and removed.
        """
        with self._service_lock:
            found = False
            if service_type in self._services:
                del self._services[service_type]
                found = True
            if service_type in self._factories:
                del self._factories[service_type]
                found = True
            if found:
                logger.debug(f"Unregistered service: {service_type.__name__}")
            return found

    def list_services(self) -> list[str]:
        """
        List all registered service type names.

        Returns:
            List of service type names.
        """
        with self._service_lock:
            types = set(self._services.keys()) | set(self._factories.keys())
            return sorted(t.__name__ for t in types)


# Module-level convenience functions


def get_service(service_type: Type[T], default: Optional[T] = None) -> T:
    """
    Get a service from the global registry.

    Args:
        service_type: The type of service to get.
        default: Default value if not found.

    Returns:
        The service instance.

    Raises:
        ServiceNotFoundError: If not found and no default provided.
    """
    return ServiceRegistry.get().resolve(service_type, default)


def register_service(service_type: Type[T], instance: T) -> None:
    """
    Register a service in the global registry.

    Args:
        service_type: The type to register under.
        instance: The service instance.
    """
    ServiceRegistry.get().register(service_type, instance)


def has_service(service_type: Type) -> bool:
    """
    Check if a service is registered.

    Args:
        service_type: The type to check.

    Returns:
        True if service is available.
    """
    return ServiceRegistry.get().has(service_type)
