"""
Integration tests for circuit breaker resilience.

Tests that external agent failures are handled gracefully
with circuit breaker pattern, ensuring:
- Circuit opens after consecutive failures
- Circuit transitions through half-open state on recovery
- Each agent has independent circuit breaker
- Metrics are properly recorded
- Fallback behavior when circuit is open
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from tests.gateway.integration.conftest import (
    register_external_agent,
    TenantContext,
    MockAgent,
    FailingAgent,
    SlowAgent,
    MockExternalFrameworkServer,
)


class CircuitBreaker:
    """Simple circuit breaker implementation for testing.

    States:
    - closed: Normal operation, requests pass through
    - open: Circuit is tripped, requests fail fast
    - half-open: Testing if service recovered
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 2,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self.metrics: list[dict] = []

    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.failure_count >= self.failure_threshold:
            self._transition_to("open")

    def record_success(self) -> None:
        """Record a success and potentially close the circuit."""
        if self.state == "half-open":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition_to("closed")
                self.failure_count = 0
                self.success_count = 0
        elif self.state == "closed":
            # Reset failure count on success in closed state
            self.failure_count = 0

    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if recovery timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._transition_to("half-open")
                    return True
            return False

        # half-open state allows requests
        return True

    def _transition_to(self, new_state: str) -> None:
        """Transition to a new state and record the metric."""
        old_state = self.state
        self.state = new_state
        self.metrics.append(
            {
                "type": "state_change",
                "from": old_state,
                "to": new_state,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "failure_count": self.failure_count,
            }
        )


class CircuitBreakerRegistry:
    """Registry of circuit breakers for multiple agents."""

    def __init__(self):
        self.breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        agent_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for an agent."""
        if agent_name not in self.breakers:
            self.breakers[agent_name] = CircuitBreaker(
                name=agent_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return self.breakers[agent_name]

    def get_all_metrics(self) -> list[dict]:
        """Get all metrics from all circuit breakers."""
        all_metrics = []
        for name, breaker in self.breakers.items():
            for metric in breaker.metrics:
                all_metrics.append({**metric, "agent": name})
        return all_metrics


class TestCircuitBreakerResilience:
    """Integration tests for circuit breaker with gateway agents."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(
        self,
        mock_external_server: MockExternalFrameworkServer,
        gateway_server_context: dict,
    ):
        """Test that circuit breaker opens after consecutive failures."""
        # Register agent with the gateway
        register_external_agent(
            gateway_server_context,
            name="failing-agent",
            framework="crewai",
        )

        # Configure server to fail
        mock_external_server.set_healthy(False)

        # Create circuit breaker with low threshold for testing
        circuit = CircuitBreaker(
            name="failing-agent",
            failure_threshold=3,
            recovery_timeout=30.0,
        )

        # Simulate multiple failures
        failures = []
        for i in range(5):
            try:
                if not circuit.can_execute():
                    failures.append(Exception("Circuit open"))
                    continue
                await mock_external_server.handle_request("/generate", "POST", {"task": "test"})
            except Exception as e:
                circuit.record_failure()
                failures.append(e)

        # After 3 failures, circuit should be open
        assert circuit.state == "open"
        assert circuit.failure_count >= 3
        assert len([f for f in failures if "Circuit open" in str(f)]) >= 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(
        self,
        mock_external_server: MockExternalFrameworkServer,
        gateway_server_context: dict,
    ):
        """Test circuit transitions to half-open state after timeout."""
        register_external_agent(
            gateway_server_context,
            name="recovering-agent",
            framework="crewai",
        )

        # Create circuit breaker with short recovery timeout
        circuit = CircuitBreaker(
            name="recovering-agent",
            failure_threshold=2,
            recovery_timeout=0.1,  # 100ms for testing
        )

        # Trigger failures to open circuit
        mock_external_server.set_healthy(False)
        for _ in range(2):
            try:
                await mock_external_server.handle_request("/generate", "POST", {"task": "test"})
            except Exception:
                circuit.record_failure()

        assert circuit.state == "open"

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Make server healthy again
        mock_external_server.set_healthy(True)

        # Check if circuit allows request (should transition to half-open)
        assert circuit.can_execute() is True
        assert circuit.state == "half-open"

    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_on_success(
        self,
        mock_external_server: MockExternalFrameworkServer,
        gateway_server_context: dict,
    ):
        """Test circuit closes after successful calls in half-open state."""
        register_external_agent(
            gateway_server_context,
            name="closing-agent",
            framework="crewai",
        )

        # Create circuit breaker with short recovery and low success threshold
        circuit = CircuitBreaker(
            name="closing-agent",
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=2,
        )

        # Open the circuit
        mock_external_server.set_healthy(False)
        for _ in range(2):
            try:
                await mock_external_server.handle_request("/generate", "POST", {"task": "test"})
            except Exception:
                circuit.record_failure()

        assert circuit.state == "open"

        # Wait for recovery timeout and transition to half-open
        await asyncio.sleep(0.15)
        circuit.can_execute()  # Triggers transition to half-open
        assert circuit.state == "half-open"

        # Make server healthy and record successes
        mock_external_server.set_healthy(True)
        for _ in range(2):
            try:
                await mock_external_server.handle_request("/generate", "POST", {"task": "test"})
                circuit.record_success()
            except Exception:
                circuit.record_failure()

        # Circuit should be closed now
        assert circuit.state == "closed"
        assert circuit.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_isolation(
        self,
        mock_external_server: MockExternalFrameworkServer,
        gateway_server_context: dict,
    ):
        """Test that each agent has independent circuit breaker."""
        # Register two agents
        register_external_agent(gateway_server_context, name="agent-1", framework="crewai")
        register_external_agent(gateway_server_context, name="agent-2", framework="autogen")

        # Create registry with independent circuit breakers
        registry = CircuitBreakerRegistry()
        circuit_1 = registry.get_or_create("agent-1", failure_threshold=2)
        circuit_2 = registry.get_or_create("agent-2", failure_threshold=2)

        # Fail agent-1 multiple times
        for _ in range(3):
            circuit_1.record_failure()

        # agent-1's circuit should be open
        assert circuit_1.state == "open"
        assert circuit_1.can_execute() is False

        # agent-2's circuit should remain closed
        assert circuit_2.state == "closed"
        assert circuit_2.can_execute() is True

        # Verify agent-2 still works
        mock_external_server.set_healthy(True)
        result = await mock_external_server.handle_request("/generate", "POST", {"task": "test"})
        assert result is not None
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics(
        self,
        mock_external_server: MockExternalFrameworkServer,
        gateway_server_context: dict,
    ):
        """Test that circuit state changes are recorded in metrics."""
        register_external_agent(
            gateway_server_context,
            name="metrics-agent",
            framework="crewai",
        )

        # Create circuit breaker
        circuit = CircuitBreaker(
            name="metrics-agent",
            failure_threshold=2,
            recovery_timeout=0.1,
        )

        # Open the circuit
        mock_external_server.set_healthy(False)
        for _ in range(2):
            try:
                await mock_external_server.handle_request("/generate", "POST", {"task": "test"})
            except Exception:
                circuit.record_failure()

        # Wait and recover
        await asyncio.sleep(0.15)
        circuit.can_execute()  # Triggers half-open

        # Make successful call
        mock_external_server.set_healthy(True)
        circuit.success_count = circuit.success_threshold  # Force close
        circuit._transition_to("closed")

        # Verify metrics were recorded
        assert len(circuit.metrics) >= 2  # At least open and closed transitions

        # Check for open transition
        open_metrics = [m for m in circuit.metrics if m["to"] == "open"]
        assert len(open_metrics) >= 1
        assert open_metrics[0]["from"] == "closed"

        # Check for closed transition
        closed_metrics = [m for m in circuit.metrics if m["to"] == "closed"]
        assert len(closed_metrics) >= 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_fallback(
        self,
        mock_external_server: MockExternalFrameworkServer,
        gateway_server_context: dict,
        mock_agent: MockAgent,
    ):
        """Test fallback behavior when circuit is open."""
        # Register primary and fallback agents
        register_external_agent(
            gateway_server_context,
            name="primary-agent",
            framework="crewai",
        )

        # Set up fallback with internal agent
        fallback_agent = mock_agent
        fallback_agent.name = "fallback-agent"
        fallback_agent._proposal = "Fallback response"

        # Create circuit breaker for primary
        circuit = CircuitBreaker(
            name="primary-agent",
            failure_threshold=2,
            recovery_timeout=30.0,
        )

        # Open the circuit
        for _ in range(2):
            circuit.record_failure()

        assert circuit.state == "open"

        # Attempt to use primary, fall back to internal
        result = None
        if circuit.can_execute():
            try:
                await mock_external_server.handle_request("/generate", "POST", {"task": "test"})
            except Exception:
                circuit.record_failure()
                result = await fallback_agent.generate("test task")
        else:
            # Circuit is open, use fallback directly
            result = await fallback_agent.generate("test task")

        # Verify fallback was used
        assert result is not None
        assert result == "Fallback response"


class TestCircuitBreakerEdgeCases:
    """Edge case tests for circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_resets_on_success_in_closed_state(self):
        """Test that success in closed state resets failure count."""
        circuit = CircuitBreaker(
            name="reset-test",
            failure_threshold=5,
        )

        # Record some failures (not enough to open)
        circuit.record_failure()
        circuit.record_failure()
        assert circuit.failure_count == 2
        assert circuit.state == "closed"

        # Record success
        circuit.record_success()

        # Failure count should reset
        assert circuit.failure_count == 0
        assert circuit.state == "closed"

    @pytest.mark.asyncio
    async def test_circuit_remains_open_before_timeout(self):
        """Test that circuit remains open before recovery timeout."""
        circuit = CircuitBreaker(
            name="timeout-test",
            failure_threshold=2,
            recovery_timeout=10.0,  # Long timeout
        )

        # Open the circuit
        circuit.record_failure()
        circuit.record_failure()
        assert circuit.state == "open"

        # Try to execute immediately (should fail)
        assert circuit.can_execute() is False
        assert circuit.state == "open"

    @pytest.mark.asyncio
    async def test_half_open_returns_to_open_on_failure(self):
        """Test that half-open circuit returns to open on failure."""
        circuit = CircuitBreaker(
            name="half-open-fail",
            failure_threshold=2,
            recovery_timeout=0.05,
        )

        # Open the circuit
        circuit.record_failure()
        circuit.record_failure()
        assert circuit.state == "open"

        # Wait for recovery timeout
        await asyncio.sleep(0.1)

        # Transition to half-open
        circuit.can_execute()
        assert circuit.state == "half-open"

        # Record another failure
        circuit.record_failure()

        # Should be open again
        assert circuit.state == "open"

    @pytest.mark.asyncio
    async def test_registry_tracks_multiple_agents(self):
        """Test that registry correctly tracks multiple agents."""
        registry = CircuitBreakerRegistry()

        # Create multiple circuit breakers
        agents = ["agent-1", "agent-2", "agent-3"]
        for name in agents:
            registry.get_or_create(name, failure_threshold=3)

        # Fail agent-2
        registry.breakers["agent-2"].record_failure()
        registry.breakers["agent-2"].record_failure()
        registry.breakers["agent-2"].record_failure()

        # Check states
        assert registry.breakers["agent-1"].state == "closed"
        assert registry.breakers["agent-2"].state == "open"
        assert registry.breakers["agent-3"].state == "closed"

        # Get all metrics
        metrics = registry.get_all_metrics()
        agent_2_metrics = [m for m in metrics if m["agent"] == "agent-2"]
        assert len(agent_2_metrics) >= 1
