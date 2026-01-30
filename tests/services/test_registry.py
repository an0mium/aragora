"""Tests for the Service Registry."""

from __future__ import annotations

import pytest

from aragora.services.registry import (
    RegistryStats,
    ServiceNotFoundError,
    ServiceRegistry,
    ServiceScope,
    get_service,
    has_service,
    register_service,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset the singleton registry before and after each test."""
    ServiceRegistry.reset()
    yield
    ServiceRegistry.reset()


class DummyService:
    """A simple service for testing."""

    def __init__(self, value: str = "default"):
        self.value = value

    def close(self):
        pass


class AnotherService:
    """Another service for testing."""

    pass


class NoCleanupService:
    """A service without cleanup methods."""

    pass


# ---------------------------------------------------------------------------
# Enum / dataclass tests
# ---------------------------------------------------------------------------


class TestRegistryDataclasses:
    def test_service_scope_values(self):
        assert ServiceScope.SINGLETON.value == "singleton"
        assert ServiceScope.TRANSIENT.value == "transient"

    def test_registry_stats_to_dict(self):
        stats = RegistryStats(
            total_services=3,
            singleton_count=2,
            transient_count=1,
            initialized_count=2,
            pending_count=1,
            total_resolves=10,
            services=[],
        )
        d = stats.to_dict()
        assert d["total_services"] == 3
        assert d["initialized_count"] == 2

    def test_service_not_found_error(self):
        err = ServiceNotFoundError(DummyService)
        assert "DummyService" in str(err)
        assert err.service_type is DummyService


# ---------------------------------------------------------------------------
# ServiceRegistry singleton
# ---------------------------------------------------------------------------


class TestServiceRegistrySingleton:
    def test_get_returns_singleton(self):
        r1 = ServiceRegistry.get()
        r2 = ServiceRegistry.get()
        assert r1 is r2

    def test_reset_clears_instance(self):
        r1 = ServiceRegistry.get()
        ServiceRegistry.reset()
        r2 = ServiceRegistry.get()
        assert r1 is not r2

    def test_reset_clears_services(self):
        reg = ServiceRegistry.get()
        reg.register(DummyService, DummyService("test"))
        ServiceRegistry.reset()
        reg2 = ServiceRegistry.get()
        assert not reg2.has(DummyService)


# ---------------------------------------------------------------------------
# register / resolve
# ---------------------------------------------------------------------------


class TestRegisterResolve:
    def test_register_and_resolve(self):
        reg = ServiceRegistry.get()
        svc = DummyService("hello")
        reg.register(DummyService, svc)
        result = reg.resolve(DummyService)
        assert result is svc
        assert result.value == "hello"

    def test_resolve_not_found_raises(self):
        reg = ServiceRegistry.get()
        with pytest.raises(ServiceNotFoundError):
            reg.resolve(DummyService)

    def test_resolve_with_default(self):
        reg = ServiceRegistry.get()
        default = DummyService("fallback")
        result = reg.resolve(DummyService, default=default)
        assert result is default

    def test_register_overwrites(self):
        reg = ServiceRegistry.get()
        reg.register(DummyService, DummyService("first"))
        reg.register(DummyService, DummyService("second"))
        result = reg.resolve(DummyService)
        assert result.value == "second"


# ---------------------------------------------------------------------------
# register_factory / lazy init
# ---------------------------------------------------------------------------


class TestRegisterFactory:
    def test_factory_lazy_init(self):
        reg = ServiceRegistry.get()
        calls = []

        def factory():
            calls.append(1)
            return DummyService("lazy")

        reg.register_factory(DummyService, factory)
        assert len(calls) == 0  # Not yet called

        result = reg.resolve(DummyService)
        assert len(calls) == 1  # Called on first resolve
        assert result.value == "lazy"

    def test_factory_cached_on_second_resolve(self):
        reg = ServiceRegistry.get()
        calls = []

        def factory():
            calls.append(1)
            return DummyService("cached")

        reg.register_factory(DummyService, factory)
        r1 = reg.resolve(DummyService)
        r2 = reg.resolve(DummyService)
        assert len(calls) == 1  # Only called once
        assert r1 is r2

    def test_register_instance_removes_factory(self):
        reg = ServiceRegistry.get()
        reg.register_factory(DummyService, lambda: DummyService("from_factory"))
        svc = DummyService("direct")
        reg.register(DummyService, svc)
        result = reg.resolve(DummyService)
        assert result.value == "direct"

    def test_register_factory_removes_instance(self):
        reg = ServiceRegistry.get()
        reg.register(DummyService, DummyService("direct"))
        reg.register_factory(DummyService, lambda: DummyService("from_factory"))
        result = reg.resolve(DummyService)
        assert result.value == "from_factory"


# ---------------------------------------------------------------------------
# has / unregister
# ---------------------------------------------------------------------------


class TestHasUnregister:
    def test_has_registered(self):
        reg = ServiceRegistry.get()
        reg.register(DummyService, DummyService())
        assert reg.has(DummyService) is True

    def test_has_not_registered(self):
        reg = ServiceRegistry.get()
        assert reg.has(DummyService) is False

    def test_has_factory(self):
        reg = ServiceRegistry.get()
        reg.register_factory(DummyService, DummyService)
        assert reg.has(DummyService) is True

    def test_unregister_instance(self):
        reg = ServiceRegistry.get()
        reg.register(DummyService, DummyService())
        found = reg.unregister(DummyService)
        assert found is True
        assert reg.has(DummyService) is False

    def test_unregister_factory(self):
        reg = ServiceRegistry.get()
        reg.register_factory(DummyService, DummyService)
        found = reg.unregister(DummyService)
        assert found is True
        assert reg.has(DummyService) is False

    def test_unregister_not_found(self):
        reg = ServiceRegistry.get()
        found = reg.unregister(DummyService)
        assert found is False


# ---------------------------------------------------------------------------
# list_services / stats
# ---------------------------------------------------------------------------


class TestListAndStats:
    def test_list_services_empty(self):
        reg = ServiceRegistry.get()
        assert reg.list_services() == []

    def test_list_services(self):
        reg = ServiceRegistry.get()
        reg.register(DummyService, DummyService())
        reg.register(AnotherService, AnotherService())
        names = reg.list_services()
        assert "AnotherService" in names
        assert "DummyService" in names
        assert names == sorted(names)

    def test_stats_empty(self):
        reg = ServiceRegistry.get()
        stats = reg.stats()
        assert stats.total_services == 0
        assert stats.initialized_count == 0

    def test_stats_with_services(self):
        reg = ServiceRegistry.get()
        reg.register(DummyService, DummyService())
        reg.register_factory(AnotherService, AnotherService)
        stats = reg.stats()
        assert stats.total_services == 2
        assert stats.initialized_count == 1
        assert stats.pending_count == 1

    def test_stats_to_dict(self):
        reg = ServiceRegistry.get()
        reg.register(DummyService, DummyService())
        d = reg.stats().to_dict()
        assert "total_services" in d
        assert d["total_services"] == 1


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_calls_close(self):
        reg = ServiceRegistry.get()
        svc = DummyService()
        calls = []
        svc.close = lambda: calls.append("closed")
        reg.register(DummyService, svc)
        hooks = reg.shutdown()
        assert hooks >= 1
        assert "closed" in calls

    def test_shutdown_skips_no_cleanup(self):
        reg = ServiceRegistry.get()
        reg.register(NoCleanupService, NoCleanupService())
        hooks = reg.shutdown()
        assert hooks == 0


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_register_and_get_service(self):
        svc = DummyService("convenience")
        register_service(DummyService, svc)
        result = get_service(DummyService)
        assert result.value == "convenience"

    def test_get_service_with_default(self):
        default = DummyService("fallback")
        result = get_service(DummyService, default=default)
        assert result is default

    def test_get_service_not_found_raises(self):
        with pytest.raises(ServiceNotFoundError):
            get_service(DummyService)

    def test_has_service_true(self):
        register_service(DummyService, DummyService())
        assert has_service(DummyService) is True

    def test_has_service_false(self):
        assert has_service(DummyService) is False
