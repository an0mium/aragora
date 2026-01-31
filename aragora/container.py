"""
Dependency Injection Container for Aragora.

Provides a lightweight, zero-dependency DI container to improve testability
by enabling dependency injection instead of direct instantiation.

This module implements:
- Container class for registering and resolving dependencies
- Protocol definitions for key injectable services
- Factory functions for creating configured instances
- Global container access via get_container()

Usage:
    from aragora.container import get_container, Container

    # Register dependencies
    container = get_container()
    container.register(BudgetCoordinatorProtocol, BudgetCoordinator)
    container.register_factory(ConvergenceDetectorProtocol, create_convergence_detector)
    container.register_instance(EventEmitterProtocol, my_emitter)

    # Resolve dependencies
    budget = container.resolve(BudgetCoordinatorProtocol)
    detector = container.resolve(ConvergenceDetectorProtocol)

Testing:
    # In tests, create isolated containers
    test_container = Container()
    test_container.register_instance(BudgetCoordinatorProtocol, MockBudgetCoordinator())

    # Or override the global container
    from aragora.container import set_container
    set_container(test_container)
"""

from __future__ import annotations

import threading
from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


# =============================================================================
# Service Protocols
# =============================================================================
# These protocols define the interfaces for key injectable services.
# They enable type-safe dependency injection and easier mocking in tests.


@runtime_checkable
class BudgetCoordinatorProtocol(Protocol):
    """Protocol for budget coordination in debates.

    Implementations handle:
    - Pre-debate budget validation
    - Mid-debate budget checks
    - Post-debate cost recording
    """

    org_id: str | None
    user_id: str | None

    def check_budget_before_debate(self, debate_id: str) -> None:
        """Check if organization has sufficient budget before starting debate.

        Raises:
            BudgetExceededError: If budget is exhausted.
        """
        ...

    def check_budget_mid_debate(self, debate_id: str, round_num: int) -> tuple[bool, str]:
        """Check if budget allows continuing the debate.

        Returns:
            Tuple of (allowed: bool, reason: str).
            allowed is True if debate can continue.
        """
        ...

    def record_debate_cost(
        self,
        debate_id: str,
        result: Any,
        token_count: int = 0,
    ) -> None:
        """Record the cost of a completed debate."""
        ...


@runtime_checkable
class ConvergenceDetectorProtocol(Protocol):
    """Protocol for semantic convergence detection.

    Implementations detect when agents' positions have converged,
    allowing early termination of debates.
    """

    def check_convergence(
        self,
        current_responses: dict[str, str],
        previous_responses: dict[str, str],
        round_number: int,
    ) -> "ConvergenceResultProtocol | None":
        """Check if debate has converged.

        Args:
            current_responses: Agent name -> response text for current round
            previous_responses: Agent name -> response text for previous round
            round_number: Current round number

        Returns:
            ConvergenceResult if enough data, None otherwise
        """
        ...


@runtime_checkable
class ConvergenceResultProtocol(Protocol):
    """Protocol for convergence check results."""

    converged: bool
    status: str
    min_similarity: float
    avg_similarity: float


@runtime_checkable
class EventEmitterProtocol(Protocol):
    """Protocol for event emission in debates.

    Implementations handle:
    - Spectator/websocket notifications
    - Moment event emission
    - Health event broadcasting
    - Agent preview emission
    """

    def notify_spectator(self, event_type: str, **data: Any) -> None:
        """Notify spectators of an event."""
        ...

    def emit_agent_preview(
        self,
        agents: list[Any],
        role_assignments: dict[Any, Any],
    ) -> None:
        """Emit agent preview for UI."""
        ...

    def emit_moment(self, moment: Any) -> None:
        """Emit a significant moment event."""
        ...


@runtime_checkable
class LifecycleManagerProtocol(Protocol):
    """Protocol for arena lifecycle management.

    Implementations handle:
    - Task cancellation
    - Cache cleanup
    - Checkpoint manager lifecycle
    - Circuit breaker metrics
    """

    async def cleanup(self) -> None:
        """Clean up resources (cancel tasks, clear caches, close managers)."""
        ...

    async def cancel_arena_tasks(self) -> None:
        """Cancel all pending arena-related asyncio tasks."""
        ...

    def clear_cache(self) -> None:
        """Clear the state cache if it exists."""
        ...

    def track_circuit_breaker_metrics(self) -> None:
        """Track circuit breaker state in metrics."""
        ...


@runtime_checkable
class OutputSanitizerProtocol(Protocol):
    """Protocol for sanitizing debate outputs.

    Implementations clean/validate outputs from agents.
    """

    @staticmethod
    def sanitize_agent_output(raw_output: str, agent_name: str) -> str:
        """Sanitize text output from an agent."""
        ...


# =============================================================================
# Container Implementation
# =============================================================================


class RegistrationError(Exception):
    """Raised when dependency registration fails."""

    pass


class ResolutionError(Exception):
    """Raised when dependency resolution fails."""

    pass


class Registration(Generic[T]):
    """Represents a registered dependency.

    Supports three registration modes:
    - type: A class to instantiate
    - factory: A callable that creates instances
    - instance: A pre-created singleton instance
    """

    def __init__(
        self,
        service_type: type[T] | None = None,
        factory: Callable[..., T] | None = None,
        instance: T | None = None,
        singleton: bool = True,
    ) -> None:
        """Initialize registration.

        Args:
            service_type: Class to instantiate (mutually exclusive with factory/instance)
            factory: Factory callable (mutually exclusive with service_type/instance)
            instance: Singleton instance (mutually exclusive with service_type/factory)
            singleton: If True, cache the created instance (default: True)
        """
        self.service_type = service_type
        self.factory = factory
        self._instance = instance
        self.singleton = singleton

        # Validate exactly one mode is set
        modes = sum([service_type is not None, factory is not None, instance is not None])
        if modes != 1:
            raise RegistrationError(
                "Exactly one of service_type, factory, or instance must be provided"
            )

    def resolve(self, container: "Container") -> T:
        """Resolve the registration to an instance.

        Args:
            container: Container for resolving nested dependencies

        Returns:
            The resolved instance
        """
        # Pre-registered instance
        if self._instance is not None:
            return self._instance

        # Factory function
        if self.factory is not None:
            instance = self.factory(container)
            if self.singleton:
                self._instance = instance
            return instance

        # Class instantiation
        if self.service_type is not None:
            instance = self.service_type()
            if self.singleton:
                self._instance = instance
            return instance

        raise ResolutionError("No resolution strategy available")


class Container:
    """Lightweight dependency injection container.

    Provides a simple, type-safe way to register and resolve dependencies.
    Supports three registration modes:

    1. Type registration - Register a class to be instantiated::

        container.register(BudgetCoordinatorProtocol, BudgetCoordinator)

    2. Factory registration - Register a factory function::

        container.register_factory(
            ConvergenceDetectorProtocol,
            lambda c: ConvergenceDetector(backend=c.resolve(SimilarityBackend))
        )

    3. Instance registration - Register a pre-created instance::

        container.register_instance(EventEmitterProtocol, my_emitter)

    Thread Safety:
        All registration and resolution operations are thread-safe.
    """

    def __init__(self) -> None:
        """Initialize empty container."""
        self._registrations: dict[type, Registration[Any]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        protocol: type[T],
        implementation: type[T],
        singleton: bool = True,
    ) -> "Container":
        """Register a class implementation for a protocol.

        Args:
            protocol: The protocol/interface type
            implementation: The implementing class
            singleton: If True, reuse the same instance (default)

        Returns:
            Self for method chaining

        Example::

            container.register(BudgetCoordinatorProtocol, BudgetCoordinator)
        """
        with self._lock:
            self._registrations[protocol] = Registration(
                service_type=implementation,
                singleton=singleton,
            )
        return self

    def register_factory(
        self,
        protocol: type[T],
        factory: Callable[["Container"], T],
        singleton: bool = True,
    ) -> "Container":
        """Register a factory function for a protocol.

        The factory receives the container for resolving nested dependencies.

        Args:
            protocol: The protocol/interface type
            factory: Callable that takes Container and returns instance
            singleton: If True, cache the created instance (default)

        Returns:
            Self for method chaining

        Example::

            container.register_factory(
                ConvergenceDetectorProtocol,
                lambda c: ConvergenceDetector(
                    threshold=0.85,
                    backend=c.resolve(SimilarityBackend)
                )
            )
        """
        with self._lock:
            self._registrations[protocol] = Registration(
                factory=factory,
                singleton=singleton,
            )
        return self

    def register_instance(
        self,
        protocol: type[T],
        instance: T,
    ) -> "Container":
        """Register a pre-created instance for a protocol.

        Args:
            protocol: The protocol/interface type
            instance: The pre-created instance

        Returns:
            Self for method chaining

        Example::

            container.register_instance(EventEmitterProtocol, my_emitter)
        """
        with self._lock:
            self._registrations[protocol] = Registration(instance=instance)
        return self

    def resolve(self, protocol: type[T]) -> T:
        """Resolve a protocol to its registered implementation.

        Args:
            protocol: The protocol/interface type to resolve

        Returns:
            The resolved instance

        Raises:
            ResolutionError: If the protocol is not registered

        Example::

            budget = container.resolve(BudgetCoordinatorProtocol)
        """
        with self._lock:
            registration = self._registrations.get(protocol)
            if registration is None:
                raise ResolutionError(
                    f"No registration found for {protocol.__name__}. "
                    f"Register it with container.register(), register_factory(), "
                    f"or register_instance()."
                )
            return registration.resolve(self)

    def try_resolve(self, protocol: type[T]) -> T | None:
        """Try to resolve a protocol, returning None if not registered.

        Args:
            protocol: The protocol/interface type to resolve

        Returns:
            The resolved instance, or None if not registered

        Example::

            budget = container.try_resolve(BudgetCoordinatorProtocol)
            if budget is not None:
                budget.check_budget_before_debate(debate_id)
        """
        try:
            return self.resolve(protocol)
        except ResolutionError:
            return None

    def is_registered(self, protocol: type) -> bool:
        """Check if a protocol is registered.

        Args:
            protocol: The protocol/interface type to check

        Returns:
            True if registered, False otherwise
        """
        with self._lock:
            return protocol in self._registrations

    def unregister(self, protocol: type) -> bool:
        """Remove a registration.

        Args:
            protocol: The protocol/interface type to unregister

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if protocol in self._registrations:
                del self._registrations[protocol]
                return True
            return False

    def clear(self) -> None:
        """Remove all registrations."""
        with self._lock:
            self._registrations.clear()

    def create_child(self) -> "Container":
        """Create a child container with current registrations copied.

        Child containers can override registrations without affecting parent.

        Returns:
            New Container with copied registrations

        Example::

            # Create test container with overrides
            test_container = container.create_child()
            test_container.register_instance(BudgetCoordinatorProtocol, mock_budget)
        """
        child = Container()
        with self._lock:
            child._registrations = dict(self._registrations)
        return child


# =============================================================================
# Global Container Management
# =============================================================================

_global_container: Container | None = None
_container_lock = threading.Lock()


def get_container() -> Container:
    """Get the global container instance.

    Creates the container on first access with default registrations.

    Returns:
        The global Container instance

    Example::

        container = get_container()
        budget = container.resolve(BudgetCoordinatorProtocol)
    """
    global _global_container
    with _container_lock:
        if _global_container is None:
            _global_container = Container()
            _configure_defaults(_global_container)
        return _global_container


def set_container(container: Container | None) -> Container | None:
    """Replace the global container.

    Useful for testing to inject a mock container.

    Args:
        container: New container, or None to reset to default

    Returns:
        The previous container

    Example::

        # In tests
        old = set_container(test_container)
        try:
            run_tests()
        finally:
            set_container(old)
    """
    global _global_container
    with _container_lock:
        old = _global_container
        _global_container = container
        return old


def reset_container() -> None:
    """Reset the global container to None.

    Next call to get_container() will create fresh container with defaults.
    """
    global _global_container
    with _container_lock:
        _global_container = None


# =============================================================================
# Default Configuration
# =============================================================================


def _configure_defaults(container: Container) -> None:
    """Configure default registrations.

    This is called automatically when the global container is first created.
    Override specific registrations in tests or for custom configurations.

    Args:
        container: Container to configure
    """
    # Import actual implementations lazily to avoid circular imports
    # and to keep the container module lightweight

    def _create_budget_coordinator(c: Container) -> BudgetCoordinatorProtocol:
        from aragora.debate.budget_coordinator import BudgetCoordinator

        return BudgetCoordinator()

    def _create_convergence_detector(c: Container) -> ConvergenceDetectorProtocol:
        from aragora.debate.convergence import ConvergenceDetector

        return ConvergenceDetector()

    def _create_event_emitter(c: Container) -> EventEmitterProtocol:
        from aragora.debate.event_emission import EventEmitter

        return EventEmitter()

    def _create_lifecycle_manager(c: Container) -> LifecycleManagerProtocol:
        from aragora.debate.lifecycle_manager import LifecycleManager

        return LifecycleManager()

    def _create_output_sanitizer(c: Container) -> OutputSanitizerProtocol:
        from aragora.debate.sanitization import OutputSanitizer

        return OutputSanitizer()

    # Register default implementations
    # These are stateless services that can be instantiated without arguments.
    # For services that require runtime configuration (PromptBuilder, JudgeSelector, etc.),
    # register them explicitly in your application or test setup.
    container.register_factory(BudgetCoordinatorProtocol, _create_budget_coordinator)
    container.register_factory(ConvergenceDetectorProtocol, _create_convergence_detector)
    container.register_factory(EventEmitterProtocol, _create_event_emitter)
    container.register_factory(LifecycleManagerProtocol, _create_lifecycle_manager)
    container.register_factory(OutputSanitizerProtocol, _create_output_sanitizer)


# =============================================================================
# Convenience Functions
# =============================================================================


def resolve(protocol: type[T]) -> T:
    """Resolve a protocol from the global container.

    Convenience function equivalent to get_container().resolve(protocol).

    Args:
        protocol: The protocol/interface type to resolve

    Returns:
        The resolved instance

    Example::

        from aragora.container import resolve

        budget = resolve(BudgetCoordinatorProtocol)
    """
    return get_container().resolve(protocol)


def try_resolve(protocol: type[T]) -> T | None:
    """Try to resolve a protocol from the global container.

    Convenience function equivalent to get_container().try_resolve(protocol).

    Args:
        protocol: The protocol/interface type to resolve

    Returns:
        The resolved instance, or None if not registered

    Example::

        from aragora.container import try_resolve

        budget = try_resolve(BudgetCoordinatorProtocol)
        if budget:
            budget.check_budget_before_debate(debate_id)
    """
    return get_container().try_resolve(protocol)
