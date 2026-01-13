"""
Tests for ServiceRegistry dependency injection system.

Tests cover:
- ServiceScope enum values
- ServiceDescriptor dataclass
- ServiceRegistry singleton behavior
- Service registration (instance and factory)
- Service resolution (singleton vs transient)
- Service unregistration and existence checks
- Shutdown hooks
- Registry statistics
- Module-level convenience functions
"""

from __future__ import annotations

import pytest
import threading
import time
from unittest.mock import Mock, MagicMock, patch

from aragora.services.registry import (
    ServiceScope,
    ServiceDescriptor,
    ServiceRegistry,
    ServiceNotFoundError,
    RegistryStats,
    get_service,
    register_service,
    has_service,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before and after each test."""
    ServiceRegistry.reset()
    yield
    ServiceRegistry.reset()


class MockService:
    """Mock service for testing."""

    def __init__(self, value: str = "default"):
        self.value = value
        self.closed = False

    def close(self):
        self.closed = True


class AnotherService:
    """Another mock service for testing multiple services."""

    def __init__(self, name: str = "another"):
        self.name = name


# ============================================================================
# ServiceScope Tests
# ============================================================================


class TestServiceScope:
    """Tests for ServiceScope enum."""

    def test_singleton_scope_value(self):
        """Test SINGLETON scope has correct value."""
        assert ServiceScope.SINGLETON.value == "singleton"

    def test_transient_scope_value(self):
        """Test TRANSIENT scope has correct value."""
        assert ServiceScope.TRANSIENT.value == "transient"

    def test_all_scopes_enumerated(self):
        """Test all expected scopes exist."""
        scopes = list(ServiceScope)
        assert len(scopes) == 2
        assert ServiceScope.SINGLETON in scopes
        assert ServiceScope.TRANSIENT in scopes


# ============================================================================
# ServiceDescriptor Tests
# ============================================================================


class TestServiceDescriptor:
    """Tests for ServiceDescriptor dataclass."""

    def test_create_with_defaults(self):
        """Test creating descriptor with minimal args."""
        descriptor = ServiceDescriptor(service_type=MockService)
        assert descriptor.service_type == MockService
        assert descriptor.scope == ServiceScope.SINGLETON
        assert descriptor.instance is None
        assert descriptor.factory is None

    def test_create_with_instance(self):
        """Test creating descriptor with instance."""
        instance = MockService("test")
        descriptor = ServiceDescriptor(service_type=MockService, instance=instance)
        assert descriptor.instance == instance

    def test_create_with_factory(self):
        """Test creating descriptor with factory."""
        factory = lambda: MockService("factory")
        descriptor = ServiceDescriptor(service_type=MockService, factory=factory)
        assert descriptor.factory == factory

    def test_to_dict_format(self):
        """Test to_dict returns expected format."""
        instance = MockService()
        descriptor = ServiceDescriptor(
            service_type=MockService, instance=instance, scope=ServiceScope.SINGLETON
        )
        result = descriptor.to_dict()

        assert result["type"] == "MockService"
        assert result["scope"] == "singleton"
        assert result["has_instance"] is True
        assert result["has_factory"] is False
        assert "registered_at" in result
        assert result["resolve_count"] == 0

    def test_resolve_count_default(self):
        """Test resolve_count defaults to 0."""
        descriptor = ServiceDescriptor(service_type=MockService)
        assert descriptor.resolve_count == 0

    def test_registered_at_timestamp(self):
        """Test registered_at is set automatically."""
        before = time.time()
        descriptor = ServiceDescriptor(service_type=MockService)
        after = time.time()

        assert before <= descriptor.registered_at <= after


# ============================================================================
# ServiceRegistry Singleton Tests
# ============================================================================


class TestServiceRegistrySingleton:
    """Tests for ServiceRegistry singleton behavior."""

    def test_get_returns_same_instance(self):
        """Test get() returns same instance on multiple calls."""
        registry1 = ServiceRegistry.get()
        registry2 = ServiceRegistry.get()
        assert registry1 is registry2

    def test_get_is_thread_safe(self):
        """Test get() is thread-safe for concurrent access."""
        registries = []
        errors = []

        def get_registry():
            try:
                registries.append(ServiceRegistry.get())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_registry) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registries) == 10
        # All should be the same instance
        assert all(r is registries[0] for r in registries)

    def test_reset_clears_all_services(self):
        """Test reset() clears all registered services."""
        registry = ServiceRegistry.get()
        registry.register(MockService, MockService())
        assert registry.has(MockService)

        ServiceRegistry.reset()

        registry = ServiceRegistry.get()
        assert not registry.has(MockService)

    def test_reset_allows_new_singleton(self):
        """Test reset() allows creating new singleton instance."""
        registry1 = ServiceRegistry.get()
        id1 = id(registry1)

        ServiceRegistry.reset()

        registry2 = ServiceRegistry.get()
        id2 = id(registry2)

        assert id1 != id2


# ============================================================================
# Service Registration Tests
# ============================================================================


class TestServiceRegistration:
    """Tests for service registration."""

    def test_register_instance(self):
        """Test registering a service instance."""
        registry = ServiceRegistry.get()
        instance = MockService("registered")

        registry.register(MockService, instance)

        assert registry.has(MockService)
        assert registry.resolve(MockService) is instance

    def test_register_overwrites_existing(self):
        """Test registering overwrites existing service."""
        registry = ServiceRegistry.get()
        instance1 = MockService("first")
        instance2 = MockService("second")

        registry.register(MockService, instance1)
        registry.register(MockService, instance2)

        resolved = registry.resolve(MockService)
        assert resolved.value == "second"

    def test_register_factory_lazy_init(self):
        """Test factory is not called until resolve."""
        registry = ServiceRegistry.get()
        factory_called = []

        def factory():
            factory_called.append(True)
            return MockService("lazy")

        registry.register_factory(MockService, factory)
        assert len(factory_called) == 0

        registry.resolve(MockService)
        assert len(factory_called) == 1

    def test_register_factory_caches_result(self):
        """Test factory result is cached after first resolve."""
        registry = ServiceRegistry.get()
        call_count = [0]

        def factory():
            call_count[0] += 1
            return MockService(f"call-{call_count[0]}")

        registry.register_factory(MockService, factory)

        result1 = registry.resolve(MockService)
        result2 = registry.resolve(MockService)

        assert result1 is result2
        assert call_count[0] == 1

    def test_register_replaces_factory(self):
        """Test direct registration replaces factory."""
        registry = ServiceRegistry.get()

        registry.register_factory(MockService, lambda: MockService("factory"))
        instance = MockService("direct")
        registry.register(MockService, instance)

        assert registry.resolve(MockService) is instance


# ============================================================================
# Service Resolution Tests
# ============================================================================


class TestServiceResolution:
    """Tests for service resolution."""

    def test_resolve_registered_service(self):
        """Test resolving a registered service."""
        registry = ServiceRegistry.get()
        instance = MockService("test")
        registry.register(MockService, instance)

        resolved = registry.resolve(MockService)
        assert resolved is instance

    def test_resolve_unregistered_raises_error(self):
        """Test resolving unregistered service raises ServiceNotFoundError."""
        registry = ServiceRegistry.get()

        with pytest.raises(ServiceNotFoundError) as exc_info:
            registry.resolve(MockService)

        assert exc_info.value.service_type == MockService

    def test_resolve_with_default(self):
        """Test resolving with default value."""
        registry = ServiceRegistry.get()
        default = MockService("default")

        resolved = registry.resolve(MockService, default=default)
        assert resolved is default

    def test_resolve_factory_creates_instance(self):
        """Test resolve creates instance from factory."""
        registry = ServiceRegistry.get()
        registry.register_factory(MockService, lambda: MockService("from-factory"))

        resolved = registry.resolve(MockService)
        assert resolved.value == "from-factory"

    def test_resolve_is_thread_safe(self):
        """Test resolve is thread-safe for concurrent access."""
        registry = ServiceRegistry.get()
        registry.register(MockService, MockService("shared"))

        results = []
        errors = []

        def resolve_service():
            try:
                results.append(registry.resolve(MockService))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=resolve_service) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert all(r is results[0] for r in results)


# ============================================================================
# Has and Unregister Tests
# ============================================================================


class TestHasAndUnregister:
    """Tests for has() and unregister() methods."""

    def test_has_returns_true_for_registered(self):
        """Test has() returns True for registered service."""
        registry = ServiceRegistry.get()
        registry.register(MockService, MockService())

        assert registry.has(MockService) is True

    def test_has_returns_true_for_factory(self):
        """Test has() returns True for factory-registered service."""
        registry = ServiceRegistry.get()
        registry.register_factory(MockService, MockService)

        assert registry.has(MockService) is True

    def test_has_returns_false_for_unregistered(self):
        """Test has() returns False for unregistered service."""
        registry = ServiceRegistry.get()

        assert registry.has(MockService) is False

    def test_unregister_removes_service(self):
        """Test unregister() removes a service."""
        registry = ServiceRegistry.get()
        registry.register(MockService, MockService())
        assert registry.has(MockService)

        result = registry.unregister(MockService)

        assert result is True
        assert registry.has(MockService) is False

    def test_unregister_removes_factory(self):
        """Test unregister() removes factory registration."""
        registry = ServiceRegistry.get()
        registry.register_factory(MockService, MockService)

        result = registry.unregister(MockService)

        assert result is True
        assert registry.has(MockService) is False

    def test_unregister_nonexistent_returns_false(self):
        """Test unregister() returns False for nonexistent service."""
        registry = ServiceRegistry.get()

        result = registry.unregister(MockService)

        assert result is False


# ============================================================================
# List Services Tests
# ============================================================================


class TestListServices:
    """Tests for list_services() method."""

    def test_list_empty_registry(self):
        """Test listing services in empty registry."""
        registry = ServiceRegistry.get()

        services = registry.list_services()

        assert services == []

    def test_list_registered_services(self):
        """Test listing registered services."""
        registry = ServiceRegistry.get()
        registry.register(MockService, MockService())
        registry.register(AnotherService, AnotherService())

        services = registry.list_services()

        assert "MockService" in services
        assert "AnotherService" in services
        assert len(services) == 2

    def test_list_includes_factories(self):
        """Test listing includes factory-registered services."""
        registry = ServiceRegistry.get()
        registry.register_factory(MockService, MockService)

        services = registry.list_services()

        assert "MockService" in services

    def test_list_sorted_alphabetically(self):
        """Test list is sorted alphabetically."""
        registry = ServiceRegistry.get()
        registry.register(MockService, MockService())
        registry.register(AnotherService, AnotherService())

        services = registry.list_services()

        assert services == sorted(services)


# ============================================================================
# Shutdown Tests
# ============================================================================


class TestShutdown:
    """Tests for shutdown() method."""

    def test_shutdown_calls_close_method(self):
        """Test shutdown calls close() on services."""
        registry = ServiceRegistry.get()
        instance = MockService()
        registry.register(MockService, instance)

        registry.shutdown()

        assert instance.closed is True

    def test_shutdown_returns_count(self):
        """Test shutdown returns number of hooks called."""
        registry = ServiceRegistry.get()
        instance = MockService()
        registry.register(MockService, instance)

        count = registry.shutdown()

        assert count == 1

    def test_shutdown_handles_missing_close(self):
        """Test shutdown handles services without close method."""
        registry = ServiceRegistry.get()

        class NoCloseService:
            pass

        registry.register(NoCloseService, NoCloseService())

        # Should not raise
        count = registry.shutdown()
        assert count == 0

    def test_shutdown_handles_close_error(self):
        """Test shutdown handles errors in close() gracefully."""
        registry = ServiceRegistry.get()

        class ErrorService:
            def close(self):
                raise RuntimeError("Close failed")

        registry.register(ErrorService, ErrorService())

        # Should not raise
        registry.shutdown()


# ============================================================================
# Stats Tests
# ============================================================================


class TestRegistryStats:
    """Tests for stats() method."""

    def test_stats_empty_registry(self):
        """Test stats on empty registry."""
        registry = ServiceRegistry.get()

        stats = registry.stats()

        assert isinstance(stats, RegistryStats)
        assert stats.total_services == 0

    def test_stats_counts_services(self):
        """Test stats counts registered services."""
        registry = ServiceRegistry.get()
        registry.register(MockService, MockService())
        registry.register(AnotherService, AnotherService())

        stats = registry.stats()

        assert stats.total_services == 2

    def test_stats_includes_factories(self):
        """Test stats includes factory-registered services."""
        registry = ServiceRegistry.get()
        registry.register_factory(MockService, MockService)

        stats = registry.stats()

        assert stats.total_services == 1


# ============================================================================
# ServiceNotFoundError Tests
# ============================================================================


class TestServiceNotFoundError:
    """Tests for ServiceNotFoundError exception."""

    def test_error_message_includes_type_name(self):
        """Test error message includes service type name."""
        error = ServiceNotFoundError(MockService)

        assert "MockService" in str(error)

    def test_error_has_service_type_attribute(self):
        """Test error has service_type attribute."""
        error = ServiceNotFoundError(MockService)

        assert error.service_type == MockService


# ============================================================================
# Module-Level Convenience Functions Tests
# ============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_service_resolves_from_global(self):
        """Test get_service() resolves from global registry."""
        instance = MockService("global")
        ServiceRegistry.get().register(MockService, instance)

        resolved = get_service(MockService)

        assert resolved is instance

    def test_get_service_with_default(self):
        """Test get_service() with default value."""
        default = MockService("default")

        resolved = get_service(MockService, default=default)

        assert resolved is default

    def test_get_service_raises_when_not_found(self):
        """Test get_service() raises when not found and no default."""
        with pytest.raises(ServiceNotFoundError):
            get_service(MockService)

    def test_register_service_adds_to_global(self):
        """Test register_service() adds to global registry."""
        instance = MockService("registered")

        register_service(MockService, instance)

        assert ServiceRegistry.get().resolve(MockService) is instance

    def test_has_service_checks_global(self):
        """Test has_service() checks global registry."""
        assert has_service(MockService) is False

        register_service(MockService, MockService())

        assert has_service(MockService) is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestServiceRegistryIntegration:
    """Integration tests for ServiceRegistry."""

    def test_full_lifecycle(self):
        """Test complete service lifecycle."""
        registry = ServiceRegistry.get()

        # Register
        instance = MockService("lifecycle")
        registry.register(MockService, instance)
        assert registry.has(MockService)

        # Resolve
        resolved = registry.resolve(MockService)
        assert resolved is instance

        # List
        services = registry.list_services()
        assert "MockService" in services

        # Stats
        stats = registry.stats()
        assert stats.total_services == 1

        # Shutdown
        registry.shutdown()
        assert instance.closed

        # Unregister
        registry.unregister(MockService)
        assert not registry.has(MockService)

    def test_multiple_services(self):
        """Test managing multiple services."""
        registry = ServiceRegistry.get()

        # Register multiple
        mock = MockService("mock")
        another = AnotherService("another")

        registry.register(MockService, mock)
        registry.register(AnotherService, another)

        # Resolve both
        assert registry.resolve(MockService) is mock
        assert registry.resolve(AnotherService) is another

        # List all
        services = registry.list_services()
        assert len(services) == 2

    def test_factory_with_dependencies(self):
        """Test factory that depends on another service."""
        registry = ServiceRegistry.get()

        # Register dependency first
        mock = MockService("dependency")
        registry.register(MockService, mock)

        # Factory that uses dependency
        def factory():
            dep = registry.resolve(MockService)
            return AnotherService(f"uses-{dep.value}")

        registry.register_factory(AnotherService, factory)

        # Resolve dependent service
        another = registry.resolve(AnotherService)
        assert another.name == "uses-dependency"
