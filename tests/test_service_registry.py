"""
Tests for the ServiceRegistry pattern.

Covers:
- Singleton pattern
- Service registration and resolution
- Factory-based lazy initialization
- Shutdown hooks
- Stats and observability
- Thread safety
- Module-level convenience functions
"""

import threading
import time
from typing import Protocol
from unittest.mock import MagicMock, patch

import pytest

from aragora.services import (
    ServiceRegistry,
    ServiceNotFoundError,
    ServiceScope,
    ServiceDescriptor,
    RegistryStats,
    get_service,
    register_service,
    has_service,
)


# =============================================================================
# Test Fixtures and Helper Classes
# =============================================================================


class MockService:
    """Simple mock service for testing."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.initialized = True


class MockServiceWithClose:
    """Mock service with close method."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class MockServiceWithShutdown:
    """Mock service with shutdown method."""

    def __init__(self):
        self.shutdown_called = False

    def shutdown(self):
        self.shutdown_called = True


class MockDatabase:
    """Mock database service."""

    def __init__(self, connection_string: str = "test://localhost"):
        self.connection_string = connection_string
        self.connected = True

    def close(self):
        self.connected = False


class MockCache:
    """Mock cache service."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value

    def clear(self):
        self.data.clear()


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the service registry before and after each test."""
    ServiceRegistry.reset()
    yield
    ServiceRegistry.reset()


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestSingletonPattern:
    """Tests for ServiceRegistry singleton behavior."""

    def test_get_returns_same_instance(self):
        """Test that get() always returns the same registry."""
        registry1 = ServiceRegistry.get()
        registry2 = ServiceRegistry.get()

        assert registry1 is registry2

    def test_reset_clears_instance(self):
        """Test that reset() clears the singleton."""
        registry1 = ServiceRegistry.get()
        registry1.register(MockService, MockService())

        ServiceRegistry.reset()

        registry2 = ServiceRegistry.get()
        assert registry2 is not registry1
        assert not registry2.has(MockService)

    def test_thread_safe_initialization(self):
        """Test that concurrent get() calls are thread-safe."""
        instances = []
        errors = []

        def get_registry():
            try:
                instances.append(ServiceRegistry.get())
            except Exception as e:
                errors.append(e)

        # Reset to force initialization
        ServiceRegistry.reset()

        # Start multiple threads simultaneously
        threads = [threading.Thread(target=get_registry) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(instances) == 10
        # All instances should be the same object
        assert all(inst is instances[0] for inst in instances)


# =============================================================================
# Service Registration Tests
# =============================================================================


class TestServiceRegistration:
    """Tests for service registration."""

    def test_register_service_instance(self):
        """Test registering a service instance."""
        registry = ServiceRegistry.get()
        service = MockService("test")

        registry.register(MockService, service)

        assert registry.has(MockService)
        assert registry.resolve(MockService) is service

    def test_register_overwrites_existing(self):
        """Test that registering overwrites existing service."""
        registry = ServiceRegistry.get()
        service1 = MockService("first")
        service2 = MockService("second")

        registry.register(MockService, service1)
        registry.register(MockService, service2)

        assert registry.resolve(MockService) is service2

    def test_register_factory(self):
        """Test registering a factory for lazy initialization."""
        registry = ServiceRegistry.get()
        factory_calls = []

        def factory():
            factory_calls.append(1)
            return MockService("factory")

        registry.register_factory(MockService, factory)

        # Factory not called yet
        assert len(factory_calls) == 0
        assert registry.has(MockService)

        # Factory called on resolve
        service = registry.resolve(MockService)
        assert len(factory_calls) == 1
        assert service.name == "factory"

        # Cached on subsequent resolves
        service2 = registry.resolve(MockService)
        assert len(factory_calls) == 1
        assert service2 is service

    def test_register_factory_with_config(self):
        """Test factory with configuration."""
        registry = ServiceRegistry.get()

        registry.register_factory(MockDatabase, lambda: MockDatabase("postgres://prod"))

        db = registry.resolve(MockDatabase)
        assert db.connection_string == "postgres://prod"


# =============================================================================
# Service Resolution Tests
# =============================================================================


class TestServiceResolution:
    """Tests for service resolution."""

    def test_resolve_registered_service(self):
        """Test resolving a registered service."""
        registry = ServiceRegistry.get()
        service = MockService()

        registry.register(MockService, service)

        assert registry.resolve(MockService) is service

    def test_resolve_missing_raises(self):
        """Test that resolving missing service raises error."""
        registry = ServiceRegistry.get()

        with pytest.raises(ServiceNotFoundError) as exc_info:
            registry.resolve(MockService)

        assert exc_info.value.service_type is MockService

    def test_resolve_with_default(self):
        """Test resolving with default value."""
        registry = ServiceRegistry.get()
        default = MockService("default")

        result = registry.resolve(MockService, default=default)

        assert result is default

    def test_resolve_with_none_default_raises(self):
        """Test resolving with None default raises ServiceNotFoundError."""
        registry = ServiceRegistry.get()

        # When default is None (the default), it raises
        with pytest.raises(ServiceNotFoundError):
            registry.resolve(MockService, default=None)


# =============================================================================
# Service Lifecycle Tests
# =============================================================================


class TestServiceLifecycle:
    """Tests for service lifecycle management."""

    def test_unregister_service(self):
        """Test unregistering a service."""
        registry = ServiceRegistry.get()
        registry.register(MockService, MockService())

        result = registry.unregister(MockService)

        assert result is True
        assert not registry.has(MockService)

    def test_unregister_missing_returns_false(self):
        """Test unregistering missing service returns False."""
        registry = ServiceRegistry.get()

        result = registry.unregister(MockService)

        assert result is False

    def test_shutdown_calls_close_methods(self):
        """Test shutdown calls close() on services."""
        registry = ServiceRegistry.get()
        service = MockServiceWithClose()

        registry.register(MockServiceWithClose, service)
        hooks_called = registry.shutdown()

        assert service.closed is True
        assert hooks_called >= 1

    def test_shutdown_calls_shutdown_methods(self):
        """Test shutdown calls shutdown() on services."""
        registry = ServiceRegistry.get()
        service = MockServiceWithShutdown()

        registry.register(MockServiceWithShutdown, service)
        hooks_called = registry.shutdown()

        assert service.shutdown_called is True
        assert hooks_called >= 1

    def test_shutdown_handles_errors(self):
        """Test shutdown handles errors gracefully."""
        registry = ServiceRegistry.get()

        class FailingService:
            def close(self):
                raise RuntimeError("Shutdown failed")

        registry.register(FailingService, FailingService())

        # Should not raise
        hooks_called = registry.shutdown()
        assert hooks_called >= 0  # May be 0 due to error


# =============================================================================
# Stats and Observability Tests
# =============================================================================


class TestRegistryStats:
    """Tests for registry statistics."""

    def test_stats_empty_registry(self):
        """Test stats on empty registry."""
        registry = ServiceRegistry.get()
        stats = registry.stats()

        assert stats.total_services == 0
        assert stats.initialized_count == 0

    def test_stats_with_services(self):
        """Test stats with registered services."""
        registry = ServiceRegistry.get()
        registry.register(MockService, MockService())
        registry.register(MockCache, MockCache())

        stats = registry.stats()

        assert stats.total_services == 2
        assert stats.initialized_count == 2
        assert len(stats.services) == 2

    def test_stats_with_factory(self):
        """Test stats with factory (not yet initialized)."""
        registry = ServiceRegistry.get()
        registry.register_factory(MockService, MockService)

        stats = registry.stats()

        assert stats.total_services == 1
        assert stats.initialized_count == 0
        assert stats.pending_count == 1

    def test_stats_to_dict(self):
        """Test stats conversion to dictionary."""
        registry = ServiceRegistry.get()
        registry.register(MockService, MockService())

        stats = registry.stats()
        d = stats.to_dict()

        assert "total_services" in d
        assert "services" in d
        assert d["total_services"] == 1

    def test_list_services(self):
        """Test listing service names."""
        registry = ServiceRegistry.get()
        registry.register(MockService, MockService())
        registry.register(MockCache, MockCache())

        services = registry.list_services()

        assert "MockService" in services
        assert "MockCache" in services


# =============================================================================
# Module-Level Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_service(self):
        """Test get_service function."""
        service = MockService("test")
        register_service(MockService, service)

        result = get_service(MockService)

        assert result is service

    def test_register_service(self):
        """Test register_service function."""
        service = MockService("test")

        register_service(MockService, service)

        assert has_service(MockService)

    def test_has_service(self):
        """Test has_service function."""
        assert not has_service(MockService)

        register_service(MockService, MockService())

        assert has_service(MockService)

    def test_get_service_with_default(self):
        """Test get_service with default."""
        default = MockService("default")

        result = get_service(MockService, default=default)

        assert result is default


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_registrations(self):
        """Test concurrent service registrations."""
        registry = ServiceRegistry.get()
        errors = []

        def register_service(i):
            try:
                registry.register(type(f"Service{i}", (), {}), MockService(f"svc-{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_service, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.list_services()) == 20

    def test_concurrent_resolve(self):
        """Test concurrent service resolution."""
        registry = ServiceRegistry.get()
        resolve_count = [0]
        lock = threading.Lock()

        def factory():
            with lock:
                resolve_count[0] += 1
            return MockService("factory")

        registry.register_factory(MockService, factory)

        results = []

        def resolve():
            results.append(registry.resolve(MockService))

        threads = [threading.Thread(target=resolve) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Factory should only be called once
        assert resolve_count[0] == 1
        # All threads should get the same instance
        assert all(r is results[0] for r in results)


# =============================================================================
# ServiceDescriptor Tests
# =============================================================================


class TestServiceDescriptor:
    """Tests for ServiceDescriptor dataclass."""

    def test_descriptor_creation(self):
        """Test creating a service descriptor."""
        descriptor = ServiceDescriptor(
            service_type=MockService,
            scope=ServiceScope.SINGLETON,
            instance=MockService(),
        )

        assert descriptor.service_type is MockService
        assert descriptor.scope == ServiceScope.SINGLETON
        assert descriptor.instance is not None

    def test_descriptor_to_dict(self):
        """Test descriptor to_dict conversion."""
        descriptor = ServiceDescriptor(
            service_type=MockService,
            scope=ServiceScope.SINGLETON,
            instance=MockService(),
        )

        d = descriptor.to_dict()

        assert d["type"] == "MockService"
        assert d["scope"] == "singleton"
        assert d["has_instance"] is True


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_register_none_instance(self):
        """Test registering None as instance."""
        registry = ServiceRegistry.get()

        # Should not raise, None is a valid value
        registry.register(MockService, None)

        # But resolve will return None
        result = registry.resolve(MockService)
        assert result is None

    def test_factory_returns_none(self):
        """Test factory that returns None."""
        registry = ServiceRegistry.get()

        registry.register_factory(MockService, lambda: None)

        result = registry.resolve(MockService)
        assert result is None

    def test_factory_raises_exception(self):
        """Test factory that raises exception."""
        registry = ServiceRegistry.get()

        def failing_factory():
            raise RuntimeError("Factory failed")

        registry.register_factory(MockService, failing_factory)

        with pytest.raises(RuntimeError, match="Factory failed"):
            registry.resolve(MockService)

    def test_resolve_after_unregister(self):
        """Test resolving after unregistering."""
        registry = ServiceRegistry.get()
        registry.register(MockService, MockService())
        registry.unregister(MockService)

        with pytest.raises(ServiceNotFoundError):
            registry.resolve(MockService)

    def test_has_with_factory_only(self):
        """Test has() returns True for factory-only service."""
        registry = ServiceRegistry.get()

        registry.register_factory(MockService, MockService)

        assert registry.has(MockService)

    def test_service_not_found_error_message(self):
        """Test ServiceNotFoundError message."""
        error = ServiceNotFoundError(MockService)

        assert "MockService" in str(error)
        assert error.service_type is MockService
