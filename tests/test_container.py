"""
Tests for the dependency injection container.

Tests cover:
- Container registration methods (type, factory, instance)
- Dependency resolution
- Singleton vs transient behavior
- Child containers
- Global container management
- Thread safety
- Error handling
"""

from __future__ import annotations

import threading
from typing import Any, Protocol, runtime_checkable
from unittest.mock import MagicMock

import pytest

from aragora.container import (
    BudgetCoordinatorProtocol,
    Container,
    ConvergenceDetectorProtocol,
    ConvergenceResultProtocol,
    EventEmitterProtocol,
    LifecycleManagerProtocol,
    OutputSanitizerProtocol,
    Registration,
    RegistrationError,
    ResolutionError,
    get_container,
    reset_container,
    resolve,
    set_container,
    try_resolve,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@runtime_checkable
class TestServiceProtocol(Protocol):
    """Test protocol for simple service."""

    name: str

    def do_something(self) -> str: ...


class TestService:
    """Simple test service implementation."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.call_count = 0

    def do_something(self) -> str:
        self.call_count += 1
        return f"done by {self.name}"


class AnotherService:
    """Another test service for dependency chains."""

    def __init__(self, test_service: TestService) -> None:
        self.test_service = test_service


@pytest.fixture
def container() -> Container:
    """Create fresh container for each test."""
    return Container()


@pytest.fixture(autouse=True)
def reset_global_container():
    """Reset global container before and after each test."""
    reset_container()
    yield
    reset_container()


# =============================================================================
# Test Registration Class
# =============================================================================


class TestRegistration:
    """Tests for Registration class."""

    def test_registration_requires_exactly_one_mode(self) -> None:
        """Test that registration requires exactly one mode."""
        # No mode - should fail
        with pytest.raises(RegistrationError, match="Exactly one"):
            Registration()

        # Multiple modes - should fail
        with pytest.raises(RegistrationError, match="Exactly one"):
            Registration(service_type=TestService, instance=TestService())

        with pytest.raises(RegistrationError, match="Exactly one"):
            Registration(service_type=TestService, factory=lambda c: TestService())

        with pytest.raises(RegistrationError, match="Exactly one"):
            Registration(instance=TestService(), factory=lambda c: TestService())

    def test_registration_with_service_type(self) -> None:
        """Test registration with service type creates instance."""
        reg: Registration[TestService] = Registration(service_type=TestService)
        container = Container()

        instance = reg.resolve(container)

        assert isinstance(instance, TestService)
        assert instance.name == "default"

    def test_registration_with_factory(self) -> None:
        """Test registration with factory calls factory."""
        call_count = 0

        def factory(c: Container) -> TestService:
            nonlocal call_count
            call_count += 1
            return TestService(name="from_factory")

        reg: Registration[TestService] = Registration(factory=factory)
        container = Container()

        instance = reg.resolve(container)

        assert isinstance(instance, TestService)
        assert instance.name == "from_factory"
        assert call_count == 1

    def test_registration_with_instance(self) -> None:
        """Test registration with instance returns same instance."""
        original = TestService(name="singleton")
        reg: Registration[TestService] = Registration(instance=original)
        container = Container()

        instance = reg.resolve(container)

        assert instance is original

    def test_singleton_caches_instance(self) -> None:
        """Test singleton registration caches created instance."""
        reg: Registration[TestService] = Registration(
            service_type=TestService,
            singleton=True,
        )
        container = Container()

        first = reg.resolve(container)
        second = reg.resolve(container)

        assert first is second

    def test_transient_creates_new_instance(self) -> None:
        """Test transient registration creates new instance each time."""
        reg: Registration[TestService] = Registration(
            service_type=TestService,
            singleton=False,
        )
        container = Container()

        first = reg.resolve(container)
        second = reg.resolve(container)

        assert first is not second


# =============================================================================
# Test Container Basic Operations
# =============================================================================


class TestContainerBasics:
    """Tests for Container basic operations."""

    def test_register_type(self, container: Container) -> None:
        """Test registering a type."""
        container.register(TestServiceProtocol, TestService)

        assert container.is_registered(TestServiceProtocol)

    def test_register_factory(self, container: Container) -> None:
        """Test registering a factory."""
        container.register_factory(
            TestServiceProtocol,
            lambda c: TestService(name="factory"),
        )

        assert container.is_registered(TestServiceProtocol)

    def test_register_instance(self, container: Container) -> None:
        """Test registering an instance."""
        instance = TestService(name="instance")
        container.register_instance(TestServiceProtocol, instance)

        assert container.is_registered(TestServiceProtocol)

    def test_register_returns_self(self, container: Container) -> None:
        """Test register methods return self for chaining."""
        result = container.register(TestServiceProtocol, TestService)
        assert result is container

        result = container.register_factory(
            TestServiceProtocol,
            lambda c: TestService(),
        )
        assert result is container

        result = container.register_instance(
            TestServiceProtocol,
            TestService(),
        )
        assert result is container

    def test_method_chaining(self, container: Container) -> None:
        """Test method chaining works."""

        @runtime_checkable
        class ServiceA(Protocol):
            pass

        @runtime_checkable
        class ServiceB(Protocol):
            pass

        class ImplA:
            pass

        class ImplB:
            pass

        container.register(ServiceA, ImplA).register(ServiceB, ImplB)

        assert container.is_registered(ServiceA)
        assert container.is_registered(ServiceB)


# =============================================================================
# Test Container Resolution
# =============================================================================


class TestContainerResolution:
    """Tests for Container resolution."""

    def test_resolve_type_registration(self, container: Container) -> None:
        """Test resolving type registration."""
        container.register(TestServiceProtocol, TestService)

        result = container.resolve(TestServiceProtocol)

        assert isinstance(result, TestService)

    def test_resolve_factory_registration(self, container: Container) -> None:
        """Test resolving factory registration."""
        container.register_factory(
            TestServiceProtocol,
            lambda c: TestService(name="from_factory"),
        )

        result = container.resolve(TestServiceProtocol)

        assert isinstance(result, TestService)
        assert result.name == "from_factory"

    def test_resolve_instance_registration(self, container: Container) -> None:
        """Test resolving instance registration."""
        original = TestService(name="original")
        container.register_instance(TestServiceProtocol, original)

        result = container.resolve(TestServiceProtocol)

        assert result is original

    def test_resolve_unregistered_raises(self, container: Container) -> None:
        """Test resolving unregistered protocol raises."""
        with pytest.raises(ResolutionError, match="No registration found"):
            container.resolve(TestServiceProtocol)

    def test_try_resolve_returns_none_if_unregistered(self, container: Container) -> None:
        """Test try_resolve returns None for unregistered."""
        result = container.try_resolve(TestServiceProtocol)

        assert result is None

    def test_try_resolve_returns_instance_if_registered(self, container: Container) -> None:
        """Test try_resolve returns instance if registered."""
        container.register(TestServiceProtocol, TestService)

        result = container.try_resolve(TestServiceProtocol)

        assert isinstance(result, TestService)


# =============================================================================
# Test Container Singleton Behavior
# =============================================================================


class TestContainerSingleton:
    """Tests for Container singleton behavior."""

    def test_singleton_default_for_type(self, container: Container) -> None:
        """Test type registration is singleton by default."""
        container.register(TestServiceProtocol, TestService)

        first = container.resolve(TestServiceProtocol)
        second = container.resolve(TestServiceProtocol)

        assert first is second

    def test_singleton_default_for_factory(self, container: Container) -> None:
        """Test factory registration is singleton by default."""
        container.register_factory(
            TestServiceProtocol,
            lambda c: TestService(),
        )

        first = container.resolve(TestServiceProtocol)
        second = container.resolve(TestServiceProtocol)

        assert first is second

    def test_transient_type_registration(self, container: Container) -> None:
        """Test transient type registration creates new instances."""
        container.register(TestServiceProtocol, TestService, singleton=False)

        first = container.resolve(TestServiceProtocol)
        second = container.resolve(TestServiceProtocol)

        assert first is not second

    def test_transient_factory_registration(self, container: Container) -> None:
        """Test transient factory registration creates new instances."""
        container.register_factory(
            TestServiceProtocol,
            lambda c: TestService(),
            singleton=False,
        )

        first = container.resolve(TestServiceProtocol)
        second = container.resolve(TestServiceProtocol)

        assert first is not second


# =============================================================================
# Test Container Dependency Chains
# =============================================================================


class TestContainerDependencyChains:
    """Tests for Container dependency chains."""

    def test_factory_can_resolve_dependencies(self, container: Container) -> None:
        """Test factory can resolve other dependencies."""
        container.register(TestServiceProtocol, TestService)

        @runtime_checkable
        class DependentProtocol(Protocol):
            test_service: TestService

        container.register_factory(
            DependentProtocol,
            lambda c: AnotherService(c.resolve(TestServiceProtocol)),
        )

        result = container.resolve(DependentProtocol)

        assert isinstance(result, AnotherService)
        assert isinstance(result.test_service, TestService)

    def test_nested_dependency_resolution(self, container: Container) -> None:
        """Test deeply nested dependency chains work."""

        @runtime_checkable
        class Level1(Protocol):
            pass

        @runtime_checkable
        class Level2(Protocol):
            pass

        @runtime_checkable
        class Level3(Protocol):
            pass

        class Impl1:
            pass

        class Impl2:
            def __init__(self, l1: Level1) -> None:
                self.l1 = l1

        class Impl3:
            def __init__(self, l2: Level2) -> None:
                self.l2 = l2

        container.register(Level1, Impl1)
        container.register_factory(Level2, lambda c: Impl2(c.resolve(Level1)))
        container.register_factory(Level3, lambda c: Impl3(c.resolve(Level2)))

        result = container.resolve(Level3)

        assert isinstance(result, Impl3)
        assert isinstance(result.l2, Impl2)
        assert isinstance(result.l2.l1, Impl1)


# =============================================================================
# Test Container Management
# =============================================================================


class TestContainerManagement:
    """Tests for Container management operations."""

    def test_unregister_removes_registration(self, container: Container) -> None:
        """Test unregister removes registration."""
        container.register(TestServiceProtocol, TestService)
        assert container.is_registered(TestServiceProtocol)

        result = container.unregister(TestServiceProtocol)

        assert result is True
        assert not container.is_registered(TestServiceProtocol)

    def test_unregister_returns_false_if_not_found(self, container: Container) -> None:
        """Test unregister returns False if not found."""
        result = container.unregister(TestServiceProtocol)

        assert result is False

    def test_clear_removes_all_registrations(self, container: Container) -> None:
        """Test clear removes all registrations."""

        @runtime_checkable
        class ServiceA(Protocol):
            pass

        @runtime_checkable
        class ServiceB(Protocol):
            pass

        class ImplA:
            pass

        class ImplB:
            pass

        container.register(ServiceA, ImplA)
        container.register(ServiceB, ImplB)

        container.clear()

        assert not container.is_registered(ServiceA)
        assert not container.is_registered(ServiceB)


# =============================================================================
# Test Child Containers
# =============================================================================


class TestChildContainers:
    """Tests for child container functionality."""

    def test_create_child_copies_registrations(self, container: Container) -> None:
        """Test child container has parent registrations."""
        container.register(TestServiceProtocol, TestService)

        child = container.create_child()

        assert child.is_registered(TestServiceProtocol)
        assert isinstance(child.resolve(TestServiceProtocol), TestService)

    def test_child_override_does_not_affect_parent(self, container: Container) -> None:
        """Test child override doesn't affect parent."""
        container.register(TestServiceProtocol, TestService)

        child = container.create_child()
        mock_service = TestService(name="mock")
        child.register_instance(TestServiceProtocol, mock_service)

        parent_instance = container.resolve(TestServiceProtocol)
        child_instance = child.resolve(TestServiceProtocol)

        assert parent_instance.name == "default"
        assert child_instance.name == "mock"

    def test_child_new_registration_does_not_affect_parent(self, container: Container) -> None:
        """Test new registration in child doesn't affect parent."""

        @runtime_checkable
        class NewProtocol(Protocol):
            pass

        class NewImpl:
            pass

        child = container.create_child()
        child.register(NewProtocol, NewImpl)

        assert child.is_registered(NewProtocol)
        assert not container.is_registered(NewProtocol)


# =============================================================================
# Test Global Container
# =============================================================================


class TestGlobalContainer:
    """Tests for global container management."""

    def test_get_container_returns_same_instance(self) -> None:
        """Test get_container returns same instance."""
        first = get_container()
        second = get_container()

        assert first is second

    def test_get_container_has_default_registrations(self) -> None:
        """Test get_container has default registrations."""
        container = get_container()

        # Should have default service protocols registered
        assert container.is_registered(BudgetCoordinatorProtocol)
        assert container.is_registered(ConvergenceDetectorProtocol)
        assert container.is_registered(EventEmitterProtocol)
        assert container.is_registered(LifecycleManagerProtocol)
        assert container.is_registered(OutputSanitizerProtocol)

    def test_set_container_replaces_global(self) -> None:
        """Test set_container replaces global container."""
        custom = Container()
        custom.register_instance(TestServiceProtocol, TestService(name="custom"))

        old = set_container(custom)

        current = get_container()
        assert current is custom
        assert current.resolve(TestServiceProtocol).name == "custom"

        # Restore
        set_container(old)

    def test_reset_container_clears_global(self) -> None:
        """Test reset_container clears global container."""
        first = get_container()
        reset_container()
        second = get_container()

        assert first is not second

    def test_resolve_convenience_function(self) -> None:
        """Test resolve convenience function."""
        # Default container should have BudgetCoordinatorProtocol
        budget = resolve(BudgetCoordinatorProtocol)
        assert budget is not None

    def test_try_resolve_convenience_function(self) -> None:
        """Test try_resolve convenience function."""
        # Should work for registered protocols
        budget = try_resolve(BudgetCoordinatorProtocol)
        assert budget is not None

        # Should return None for unregistered
        result = try_resolve(TestServiceProtocol)
        assert result is None


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_registration(self, container: Container) -> None:
        """Test concurrent registration is thread-safe."""
        protocols = []
        for i in range(10):

            @runtime_checkable
            class DynamicProtocol(Protocol):
                pass

            DynamicProtocol.__name__ = f"Protocol{i}"
            protocols.append(DynamicProtocol)

        def register_protocol(proto: type, idx: int) -> None:
            class DynamicImpl:
                value = idx

            container.register(proto, DynamicImpl)

        threads = [
            threading.Thread(target=register_protocol, args=(p, i)) for i, p in enumerate(protocols)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be registered
        for proto in protocols:
            assert container.is_registered(proto)

    def test_concurrent_resolution(self, container: Container) -> None:
        """Test concurrent resolution is thread-safe."""
        counter = {"value": 0}
        lock = threading.Lock()

        def counting_factory(c: Container) -> TestService:
            with lock:
                counter["value"] += 1
            return TestService(name=f"instance_{counter['value']}")

        container.register_factory(TestServiceProtocol, counting_factory, singleton=True)

        results = []
        errors = []

        def resolve_service() -> None:
            try:
                result = container.resolve(TestServiceProtocol)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=resolve_service) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 10
        # All should be same instance due to singleton
        first = results[0]
        for r in results[1:]:
            assert r is first


# =============================================================================
# Test Protocol Definitions
# =============================================================================


class TestProtocolDefinitions:
    """Tests for protocol definitions in container module."""

    def test_budget_coordinator_protocol(self) -> None:
        """Test BudgetCoordinatorProtocol can be implemented."""

        class MockBudgetCoordinator:
            org_id: str | None = None
            user_id: str | None = None

            def check_budget_before_debate(self, debate_id: str) -> None:
                pass

            def check_budget_mid_debate(self, debate_id: str, round_num: int) -> bool:
                return True

            def record_debate_cost(self, debate_id: str, result: Any, token_count: int = 0) -> None:
                pass

        mock = MockBudgetCoordinator()
        assert isinstance(mock, BudgetCoordinatorProtocol)

    def test_convergence_detector_protocol(self) -> None:
        """Test ConvergenceDetectorProtocol can be implemented."""

        class MockResult:
            converged = True
            similarity = 0.9
            details: dict[str, Any] = {}

        class MockConvergenceDetector:
            def check_convergence(
                self, positions: list[str], threshold: float = 0.85
            ) -> MockResult:
                return MockResult()

            def get_similarity(self, text_a: str, text_b: str) -> float:
                return 0.9

        mock = MockConvergenceDetector()
        assert isinstance(mock, ConvergenceDetectorProtocol)

    def test_event_emitter_protocol(self) -> None:
        """Test EventEmitterProtocol can be implemented."""

        class MockEventEmitter:
            def notify_spectator(self, event_type: str, **data: Any) -> None:
                pass

            def emit_agent_preview(
                self,
                agents: list[Any],
                role_assignments: dict[str, str] | None = None,
            ) -> None:
                pass

            def emit_moment(self, moment_type: str, data: dict[str, Any]) -> None:
                pass

        mock = MockEventEmitter()
        assert isinstance(mock, EventEmitterProtocol)

    def test_lifecycle_manager_protocol(self) -> None:
        """Test LifecycleManagerProtocol can be implemented."""

        class MockLifecycleManager:
            async def cleanup(self) -> None:
                pass

            async def __aenter__(self) -> "MockLifecycleManager":
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: Any,
            ) -> bool | None:
                return None

        mock = MockLifecycleManager()
        assert isinstance(mock, LifecycleManagerProtocol)

    def test_output_sanitizer_protocol(self) -> None:
        """Test OutputSanitizerProtocol can be implemented."""

        class MockOutputSanitizer:
            @staticmethod
            def sanitize_agent_output(raw_output: str, agent_name: str) -> str:
                return raw_output.strip()

        mock = MockOutputSanitizer()
        assert isinstance(mock, OutputSanitizerProtocol)


# =============================================================================
# Test Default Container Configuration
# =============================================================================


class TestDefaultConfiguration:
    """Tests for default container configuration."""

    def test_default_budget_coordinator_resolves(self) -> None:
        """Test default BudgetCoordinator can be resolved."""
        container = get_container()

        budget = container.resolve(BudgetCoordinatorProtocol)

        assert budget is not None
        assert hasattr(budget, "check_budget_before_debate")

    def test_default_convergence_detector_resolves(self) -> None:
        """Test default ConvergenceDetector can be resolved."""
        container = get_container()

        detector = container.resolve(ConvergenceDetectorProtocol)

        assert detector is not None
        assert hasattr(detector, "check_convergence")

    def test_default_event_emitter_resolves(self) -> None:
        """Test default EventEmitter can be resolved."""
        container = get_container()

        emitter = container.resolve(EventEmitterProtocol)

        assert emitter is not None
        assert hasattr(emitter, "notify_spectator")

    def test_default_lifecycle_manager_resolves(self) -> None:
        """Test default LifecycleManager can be resolved."""
        container = get_container()

        manager = container.resolve(LifecycleManagerProtocol)

        assert manager is not None
        assert hasattr(manager, "cleanup")

    def test_default_output_sanitizer_resolves(self) -> None:
        """Test default OutputSanitizer can be resolved."""
        container = get_container()

        sanitizer = container.resolve(OutputSanitizerProtocol)

        assert sanitizer is not None
        assert hasattr(sanitizer, "sanitize_agent_output")


# =============================================================================
# Test Usage Patterns
# =============================================================================


class TestUsagePatterns:
    """Tests demonstrating common usage patterns."""

    def test_testing_pattern_with_mock(self) -> None:
        """Test pattern for using mocks in tests."""
        # Create test container
        test_container = Container()

        # Register mock
        mock_budget = MagicMock(spec=BudgetCoordinatorProtocol)
        mock_budget.org_id = "test-org"
        mock_budget.user_id = "test-user"
        test_container.register_instance(BudgetCoordinatorProtocol, mock_budget)

        # Use in test
        budget = test_container.resolve(BudgetCoordinatorProtocol)
        budget.check_budget_before_debate("debate-123")

        mock_budget.check_budget_before_debate.assert_called_once_with("debate-123")

    def test_override_global_for_tests(self) -> None:
        """Test pattern for overriding global container in tests."""
        # Create custom container
        test_container = Container()
        mock_emitter = MagicMock(spec=EventEmitterProtocol)
        test_container.register_instance(EventEmitterProtocol, mock_emitter)

        # Override global
        old = set_container(test_container)
        try:
            # Code under test uses global container
            emitter = resolve(EventEmitterProtocol)
            emitter.notify_spectator("test_event", data="test")

            mock_emitter.notify_spectator.assert_called_once()
        finally:
            # Always restore
            set_container(old)

    def test_child_container_for_request_scope(self) -> None:
        """Test pattern for request-scoped dependencies."""
        # Main container with singleton services
        main = get_container()

        # Request-scoped child with per-request overrides
        request_container = main.create_child()
        request_budget = MagicMock(spec=BudgetCoordinatorProtocol)
        request_budget.org_id = "request-org"
        request_container.register_instance(BudgetCoordinatorProtocol, request_budget)

        # Resolve from request container
        budget = request_container.resolve(BudgetCoordinatorProtocol)

        assert budget.org_id == "request-org"

        # Main container still has original
        main_budget = main.resolve(BudgetCoordinatorProtocol)
        assert main_budget is not request_budget
