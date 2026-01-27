"""
Lazy Store Factory for thread-safe, lazy-initialized stores.

This module provides a generic factory pattern for creating lazy-initialized
stores throughout the handlers codebase, eliminating duplicate boilerplate.

Usage:
    # Define a store factory at module level
    email_store = LazyStoreFactory(
        store_name="email_store",
        import_path="aragora.storage.email_store",
        factory_name="get_email_store",
        logger_context="SharedInbox",
    )

    # Use it in your handler
    store = email_store.get()
    if store:
        store.do_something()
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class LazyStoreFactory:
    """
    Thread-safe factory for lazy-initialized stores.

    Provides double-checked locking pattern with comprehensive error handling
    for importing and initializing store instances.

    Attributes:
        store_name: Name of the store for logging
        import_path: Full module path to import (e.g., "aragora.storage.email_store")
        factory_name: Name of the factory function in the module (e.g., "get_email_store")
        logger_context: Context string for log messages (e.g., "SharedInbox")
        factory_args: Optional positional arguments for the factory function
        factory_kwargs: Optional keyword arguments for the factory function
    """

    def __init__(
        self,
        store_name: str,
        import_path: str,
        factory_name: str,
        logger_context: str = "Handler",
        factory_args: Optional[tuple] = None,
        factory_kwargs: Optional[dict] = None,
    ):
        self.store_name = store_name
        self.import_path = import_path
        self.factory_name = factory_name
        self.logger_context = logger_context
        self.factory_args = factory_args or ()
        self.factory_kwargs = factory_kwargs or {}

        self._store: Optional[Any] = None
        self._lock = threading.Lock()
        self._initialized = False
        self._init_error: Optional[str] = None

    def get(self) -> Optional[Any]:
        """
        Get the store instance, initializing lazily if needed.

        Uses double-checked locking for thread safety with minimal contention.

        Returns:
            The store instance, or None if initialization failed.
        """
        # Fast path: already initialized
        if self._initialized:
            return self._store

        # Slow path: acquire lock and initialize
        with self._lock:
            # Double-check after acquiring lock
            if self._initialized:
                return self._store

            self._store = self._initialize()
            self._initialized = True

        return self._store

    def _initialize(self) -> Optional[Any]:
        """
        Initialize the store by importing and calling the factory.

        Returns:
            The initialized store, or None if initialization failed.
        """
        try:
            module = importlib.import_module(self.import_path)
            factory: Callable[..., Any] = getattr(module, self.factory_name)
            store = factory(*self.factory_args, **self.factory_kwargs)
            logger.info(f"[{self.logger_context}] Initialized {self.store_name}")
            return store

        except ImportError as e:
            self._init_error = f"Module not available: {e}"
            logger.warning(f"[{self.logger_context}] {self.store_name} module not available: {e}")
            return None

        except AttributeError as e:
            self._init_error = f"Factory function not found: {e}"
            logger.warning(f"[{self.logger_context}] {self.store_name} factory not found: {e}")
            return None

        except (OSError, IOError, RuntimeError) as e:
            self._init_error = f"Init failed: {type(e).__name__}: {e}"
            logger.warning(
                f"[{self.logger_context}] {self.store_name} init failed: {type(e).__name__}: {e}"
            )
            return None

        except Exception as e:
            self._init_error = f"Unexpected error: {e}"
            logger.warning(f"[{self.logger_context}] Failed to init {self.store_name}: {e}")
            return None

    def reset(self) -> None:
        """
        Reset the factory, allowing re-initialization.

        Useful for testing or recovering from transient failures.
        """
        with self._lock:
            self._store = None
            self._initialized = False
            self._init_error = None

    @property
    def is_initialized(self) -> bool:
        """Check if the store has been initialized (successfully or not)."""
        return self._initialized

    @property
    def is_available(self) -> bool:
        """Check if the store is initialized and available."""
        return self._initialized and self._store is not None

    @property
    def initialization_error(self) -> Optional[str]:
        """Get the error message if initialization failed."""
        return self._init_error


class LazyStoreRegistry:
    """
    Registry for managing multiple LazyStoreFactory instances.

    Useful for handlers that need multiple stores with a consistent interface.

    Usage:
        stores = LazyStoreRegistry()
        stores.register("email", "aragora.storage.email_store", "get_email_store", "Inbox")
        stores.register("rules", "aragora.services.rules_store", "get_rules_store", "Inbox")

        email_store = stores.get("email")
        rules_store = stores.get("rules")
    """

    def __init__(self):
        self._factories: dict[str, LazyStoreFactory] = {}

    def register(
        self,
        name: str,
        import_path: str,
        factory_name: str,
        logger_context: str = "Handler",
        factory_args: Optional[tuple] = None,
        factory_kwargs: Optional[dict] = None,
    ) -> LazyStoreFactory:
        """
        Register a new store factory.

        Args:
            name: Unique name for this store
            import_path: Full module path to import
            factory_name: Name of the factory function
            logger_context: Context for log messages
            factory_args: Optional positional args for factory
            factory_kwargs: Optional keyword args for factory

        Returns:
            The created LazyStoreFactory instance
        """
        factory = LazyStoreFactory(
            store_name=name,
            import_path=import_path,
            factory_name=factory_name,
            logger_context=logger_context,
            factory_args=factory_args,
            factory_kwargs=factory_kwargs,
        )
        self._factories[name] = factory
        return factory

    def get(self, name: str) -> Optional[Any]:
        """
        Get a store by name.

        Args:
            name: The registered name of the store

        Returns:
            The store instance, or None if not registered or init failed
        """
        factory = self._factories.get(name)
        if factory is None:
            logger.warning(f"Store '{name}' not registered")
            return None
        return factory.get()

    def get_factory(self, name: str) -> Optional[LazyStoreFactory]:
        """Get the factory instance by name."""
        return self._factories.get(name)

    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset store(s), allowing re-initialization.

        Args:
            name: If provided, reset only this store. Otherwise reset all.
        """
        if name is not None:
            factory = self._factories.get(name)
            if factory:
                factory.reset()
        else:
            for factory in self._factories.values():
                factory.reset()

    def reset_all(self) -> None:
        """Reset all registered stores."""
        self.reset()

    @property
    def registered_stores(self) -> list[str]:
        """List of registered store names."""
        return list(self._factories.keys())

    def status(self) -> dict[str, dict[str, Any]]:
        """
        Get status of all registered stores.

        Returns:
            Dict mapping store names to status info
        """
        return {
            name: {
                "initialized": factory.is_initialized,
                "available": factory.is_available,
                "error": factory.initialization_error,
            }
            for name, factory in self._factories.items()
        }
