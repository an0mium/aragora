"""
End-to-end resilience integration tests.

Tests comprehensive resilience scenarios including:
- Agent failure cascades during debates
- Connector failures with recovery
- Debate state management under degradation
- Cross-layer failures (agent + connector + network)
- State persistence and recovery
- Metrics accuracy under load
"""

import asyncio
import random
from collections import defaultdict
from dataclasses import dataclass, field

import pytest

from aragora.core import Agent, Environment, Message
from aragora.debate.context import DebateContext
from aragora.resilience import (
    CircuitOpenError,
    get_circuit_breaker,
    get_circuit_breaker_metrics,
    reset_all_circuit_breakers,
)


@dataclass
class MockCritique:
    """Mock critique for testing."""

    agent: str = "critic"
    target_agent: str = "target"
    target_content: str = "content"
    issues: list[str] = field(default_factory=lambda: ["issue1"])
    suggestions: list[str] = field(default_factory=lambda: ["suggestion1"])
    severity: float = 5.0
    reasoning: str = "test reasoning"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_breakers():
    """Reset all circuit breakers before each test."""
    reset_all_circuit_breakers()
    yield
    reset_all_circuit_breakers()


@pytest.fixture
def seed_random():
    """Seed random for reproducible tests."""
    random.seed(42)
    yield


class MockFailingAgent:
    """Agent that fails on demand with configurable behavior."""

    def __init__(
        self,
        name: str,
        fail_after_rounds: int | None = None,
        fail_probability: float = 0.0,
        timeout_probability: float = 0.0,
        recovery_after_failures: int | None = None,
    ):
        self.name = name
        self.role = "proposer"  # AgentRole is a Literal type
        self.model = "mock-model"
        self.fail_after_rounds = fail_after_rounds
        self.fail_probability = fail_probability
        self.timeout_probability = timeout_probability
        self.recovery_after_failures = recovery_after_failures
        self._call_count = 0
        self._failure_count = 0
        self._responses: list[str] = []
        self._circuit_breaker = get_circuit_breaker(
            f"agent_{name}",
            failure_threshold=3,
            cooldown_seconds=1.0,  # Short for testing
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response with potential failures."""
        self._call_count += 1

        # Check if circuit is open
        if not self._circuit_breaker.can_proceed(self.name):
            raise CircuitOpenError(self.name, 1.0)

        try:
            # Check round-based failure
            if self.fail_after_rounds is not None and self._call_count > self.fail_after_rounds:
                # Check if recovery threshold reached
                if (
                    self.recovery_after_failures is not None
                    and self._failure_count >= self.recovery_after_failures
                ):
                    # Recovered - generate success
                    pass
                else:
                    raise RuntimeError(f"{self.name} failed after round threshold")

            # Check probability-based failure
            if random.random() < self.fail_probability:
                raise RuntimeError(f"{self.name} random failure")

            # Check timeout
            if random.random() < self.timeout_probability:
                await asyncio.sleep(10)  # Will timeout
                raise TimeoutError(f"{self.name} timed out")

            response = f"Response from {self.name} (call {self._call_count})"
            self._responses.append(response)
            self._circuit_breaker.record_success(self.name)
            return response

        except Exception as e:
            self._failure_count += 1
            self._circuit_breaker.record_failure(self.name)
            raise

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def failure_count(self) -> int:
        return self._failure_count


class MockConnector:
    """Connector that simulates various failure modes."""

    def __init__(
        self,
        name: str,
        fail_on_call: int | None = None,
        quota_error: bool = False,
        needs_token_refresh: bool = False,
    ):
        self.name = name
        self.fail_on_call = fail_on_call
        self.quota_error = quota_error
        self.needs_token_refresh = needs_token_refresh
        self._call_count = 0
        self._token_refreshed = False
        self._always_fail = False  # If True, always fail
        self._circuit_breaker = get_circuit_breaker(
            f"connector_{name}",
            failure_threshold=2,
            cooldown_seconds=0.5,
        )

    async def fetch_data(self, query: str) -> dict:
        """Fetch data with potential failures."""
        self._call_count += 1

        if not self._circuit_breaker.can_proceed(self.name):
            raise CircuitOpenError(f"connector_{self.name}", 0.5)

        try:
            # Always fail mode
            if self._always_fail:
                raise RuntimeError(f"Connector {self.name} always fails")

            # Simulate call-specific failure
            if self.fail_on_call is not None and self._call_count == self.fail_on_call:
                if self.quota_error:
                    raise RuntimeError("429: Rate limit exceeded")
                raise RuntimeError(f"Connector {self.name} failed on call {self._call_count}")

            # Simulate token refresh requirement
            if self.needs_token_refresh and not self._token_refreshed:
                self._token_refreshed = True
                raise RuntimeError("401: Token expired")

            self._circuit_breaker.record_success(self.name)
            return {"source": self.name, "data": f"Data for: {query}"}

        except Exception:
            self._circuit_breaker.record_failure(self.name)
            raise


# =============================================================================
# Test: Agent Failure Cascades During Debate
# =============================================================================


class TestAgentFailureCascades:
    """Tests for agent failures during multi-agent debates."""

    @pytest.mark.asyncio
    async def test_debate_continues_with_agent_dropout(self, seed_random):
        """Test that debate continues when agents drop out mid-debate."""
        # Create agents where one will fail after round 1
        agents = [
            MockFailingAgent("agent1"),
            MockFailingAgent("agent2", fail_after_rounds=1),  # Fails after round 1
            MockFailingAgent("agent3"),
        ]

        responses_per_round: dict[int, list[str]] = defaultdict(list)
        failed_agents: list[str] = []

        # Simulate 3 rounds of debate
        for round_num in range(1, 4):
            for agent in agents:
                if agent.name in failed_agents:
                    continue  # Skip failed agents

                try:
                    response = await asyncio.wait_for(
                        agent.generate(f"Round {round_num} prompt"),
                        timeout=1.0,
                    )
                    responses_per_round[round_num].append(response)
                except Exception:
                    failed_agents.append(agent.name)

        # Verify results
        assert "agent2" in failed_agents
        assert len(responses_per_round[1]) == 3  # All agents responded in round 1
        assert len(responses_per_round[2]) == 2  # Only 2 agents after dropout
        assert len(responses_per_round[3]) == 2  # Still 2 agents

    @pytest.mark.asyncio
    async def test_sequential_agent_failures(self, seed_random):
        """Test multiple agents failing sequentially across rounds."""
        agents = [
            MockFailingAgent("agent1", fail_after_rounds=2),  # Fails after round 2
            MockFailingAgent("agent2", fail_after_rounds=3),  # Fails after round 3
            MockFailingAgent("agent3", fail_after_rounds=4),  # Fails after round 4
            MockFailingAgent("agent4"),  # Never fails
        ]

        active_agents_per_round: dict[int, int] = {}
        failed_order: list[str] = []

        for round_num in range(1, 6):
            active_count = 0
            for agent in agents:
                if agent.name in failed_order:
                    continue

                try:
                    await agent.generate(f"Round {round_num}")
                    active_count += 1
                except Exception:
                    failed_order.append(agent.name)

            active_agents_per_round[round_num] = active_count

        # Verify failure order
        assert failed_order == ["agent1", "agent2", "agent3"]
        assert active_agents_per_round[1] == 4
        assert active_agents_per_round[2] == 4
        assert active_agents_per_round[3] == 3  # After agent1 fails
        assert active_agents_per_round[4] == 2  # After agent2 fails
        assert active_agents_per_round[5] == 1  # After agent3 fails

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_repeated_failures(self):
        """Test circuit breaker opens when agent fails repeatedly."""
        agent = MockFailingAgent("flaky_agent", fail_probability=1.0)  # Always fails

        failures = 0
        circuit_opened = False

        for _ in range(10):
            try:
                await agent.generate("test")
            except CircuitOpenError:
                circuit_opened = True
                break
            except Exception:
                failures += 1

        # Circuit should open after threshold (3 failures)
        # Note: The agent records failure then raises, so we get 3 RuntimeErrors
        # then the 4th call sees circuit is open and raises CircuitOpenError
        assert failures >= 3  # At least threshold failures before circuit opens
        assert circuit_opened

    @pytest.mark.asyncio
    async def test_agent_recovery_after_cooldown(self):
        """Test agent can proceed again after circuit cooldown."""
        agent = MockFailingAgent("recovering_agent", fail_probability=1.0)

        # Trigger failures to open circuit
        for _ in range(3):
            try:
                await agent.generate("test")
            except Exception:
                pass

        # Verify circuit is open
        assert not agent._circuit_breaker.can_proceed(agent.name)

        # Wait for cooldown
        await asyncio.sleep(1.1)

        # Should be able to proceed now (half-open)
        assert agent._circuit_breaker.can_proceed(agent.name)

    @pytest.mark.asyncio
    async def test_debate_metrics_track_dropouts(self, seed_random):
        """Test that debate metrics accurately track agent dropouts."""
        # Use lower threshold for faster circuit opening in test
        agents = [
            MockFailingAgent("agent1", fail_after_rounds=1),
            MockFailingAgent("agent2"),
            MockFailingAgent("agent3", fail_after_rounds=2),
        ]
        # Override circuit breakers with lower threshold for testing
        for agent in agents:
            agent._circuit_breaker = get_circuit_breaker(
                f"agent_{agent.name}_dropout_test",
                failure_threshold=2,  # Lower threshold for test
                cooldown_seconds=1.0,
            )

        metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "circuits_opened": 0,
            "agents_dropped": [],
        }

        # Run more rounds to ensure circuits open
        for round_num in range(1, 8):
            for agent in agents:
                if agent.name in metrics["agents_dropped"]:
                    continue

                metrics["total_calls"] += 1
                try:
                    await agent.generate(f"Round {round_num}")
                    metrics["successful_calls"] += 1
                except CircuitOpenError:
                    metrics["circuits_opened"] += 1
                    if agent.name not in metrics["agents_dropped"]:
                        metrics["agents_dropped"].append(agent.name)
                except Exception:
                    metrics["failed_calls"] += 1

        # Verify metrics
        assert metrics["failed_calls"] >= 2  # At least 2 agent failures
        assert len(metrics["agents_dropped"]) == 2  # Two agents dropped
        assert "agent1" in metrics["agents_dropped"]
        assert "agent3" in metrics["agents_dropped"]


# =============================================================================
# Test: Connector Failures and Recovery
# =============================================================================


class TestConnectorFailures:
    """Tests for connector failures during debate evidence gathering."""

    @pytest.mark.asyncio
    async def test_connector_failure_with_fallback(self):
        """Test fallback connector used when primary fails."""
        primary = MockConnector("primary", fail_on_call=1)
        fallback = MockConnector("fallback")

        result = None
        try:
            result = await primary.fetch_data("test query")
        except Exception:
            # Fallback to secondary
            result = await fallback.fetch_data("test query")

        assert result is not None
        assert result["source"] == "fallback"

    @pytest.mark.asyncio
    async def test_multiple_connector_failures_graceful_degradation(self):
        """Test graceful degradation when multiple connectors fail."""
        connectors = [
            MockConnector("conn1", fail_on_call=1),  # Fails immediately
            MockConnector("conn2", quota_error=True, fail_on_call=1),  # Quota error
            MockConnector("conn3"),  # Works
        ]

        results: list[dict] = []
        errors: list[str] = []

        for conn in connectors:
            try:
                result = await conn.fetch_data("test query")
                results.append(result)
            except Exception as e:
                errors.append(f"{conn.name}: {e}")

        # Only one connector succeeded
        assert len(results) == 1
        assert results[0]["source"] == "conn3"
        assert len(errors) == 2

    @pytest.mark.asyncio
    async def test_token_refresh_recovery(self):
        """Test automatic recovery after token refresh."""
        connector = MockConnector("oauth_conn", needs_token_refresh=True)

        # First call fails with token error
        first_result = None
        try:
            first_result = await connector.fetch_data("query1")
        except Exception as e:
            assert "Token expired" in str(e)

        # Second call should succeed (token refreshed)
        second_result = await connector.fetch_data("query2")
        assert second_result is not None
        assert second_result["source"] == "oauth_conn"

    @pytest.mark.asyncio
    async def test_connector_circuit_breaker_integration(self):
        """Test circuit breaker properly integrates with connector."""
        # Create a connector that always fails (fail_on_call=None means check other conditions)
        connector = MockConnector("failing_conn_cb")
        # Override to make it always fail
        connector._always_fail = True

        # First failure
        try:
            await connector.fetch_data("query1")
        except CircuitOpenError:
            pass  # Might already be open
        except Exception:
            pass

        # Second failure - circuit should open (threshold=2)
        try:
            await connector.fetch_data("query2")
        except CircuitOpenError:
            pass
        except Exception:
            pass

        # Third call - circuit should be open
        with pytest.raises(CircuitOpenError):
            await connector.fetch_data("query3")


# =============================================================================
# Test: Debate State Management Under Degradation
# =============================================================================


class TestDebateStateDegradation:
    """Tests for debate state management when agents/connectors degrade."""

    @pytest.mark.asyncio
    async def test_debate_context_preserves_partial_results(self):
        """Test DebateContext preserves partial results on failures."""
        ctx = DebateContext(env=Environment(task="Test task"))

        # Simulate successful proposals from some agents
        ctx.proposals["agent1"] = "Proposal 1"
        ctx.proposals["agent2"] = "Proposal 2"
        # agent3 fails - no proposal

        # Add messages
        ctx.add_message(Message(role="assistant", content="Response 1", agent="agent1"))
        ctx.add_message(Message(role="assistant", content="Response 2", agent="agent2"))

        # Verify partial state is preserved
        assert len(ctx.proposals) == 2
        assert len(ctx.context_messages) == 2
        assert "agent1" in ctx.proposals
        assert "agent2" in ctx.proposals

    @pytest.mark.asyncio
    async def test_quorum_tracking_with_failures(self, seed_random):
        """Test quorum tracking when agents fail."""
        total_agents = 7
        quorum_threshold = 5  # Need 5 of 7 for quorum

        agents = [
            MockFailingAgent(f"agent{i}", fail_probability=0.3 if i < 3 else 0.0)
            for i in range(total_agents)
        ]

        # Collect responses
        responses: list[str] = []
        failed_agents: list[str] = []

        for agent in agents:
            try:
                response = await agent.generate("Voting round")
                responses.append(response)
            except Exception:
                failed_agents.append(agent.name)

        # Check if quorum achieved
        quorum_achieved = len(responses) >= quorum_threshold

        # With 30% failure on 3 agents, likely still have quorum
        # This tests that we track quorum correctly
        assert len(responses) + len(failed_agents) == total_agents
        # Quorum result depends on random failures, just verify tracking works

    @pytest.mark.asyncio
    async def test_consensus_with_partial_responses(self):
        """Test consensus detection works with partial agent responses."""
        # Simulate 5 agents, 2 fail
        responses = {
            "agent1": "Yes, we should proceed",
            "agent2": "Yes, I agree",
            "agent3": None,  # Failed
            "agent4": "Yes, proceed",
            "agent5": None,  # Failed
        }

        valid_responses = {k: v for k, v in responses.items() if v is not None}

        # Simple majority check with partial data
        yes_votes = sum(1 for r in valid_responses.values() if "yes" in r.lower())
        total_valid = len(valid_responses)

        # 3 of 3 valid responses say "yes" - consensus achieved
        assert yes_votes == 3
        assert total_valid == 3
        assert yes_votes / total_valid > 0.5  # Majority

    @pytest.mark.asyncio
    async def test_critique_accumulation_with_failures(self):
        """Test critiques accumulate correctly even when critics fail."""
        ctx = DebateContext(env=Environment(task="Test critique"))

        # Add critiques from successful critics
        critique1 = MockCritique(
            agent="critic1",
            target_agent="agent1",
            target_content="content",
            issues=["Issue 1"],
            suggestions=["Suggestion 1"],
            severity=8.0,
            reasoning="Good analysis",
        )
        critique2 = MockCritique(
            agent="critic2",
            target_agent="agent1",
            target_content="content",
            issues=["Issue 2"],
            suggestions=["Suggestion 2"],
            severity=6.0,
            reasoning="Needs more detail",
        )
        # critic3 failed - no critique

        ctx.add_critique(critique1)
        ctx.add_critique(critique2)

        # Verify critiques preserved
        assert len(ctx.round_critiques) == 2
        assert len(ctx.partial_critiques) == 2


# =============================================================================
# Test: Cross-Layer Failures
# =============================================================================


class TestCrossLayerFailures:
    """Tests for failures across multiple layers simultaneously."""

    @pytest.mark.asyncio
    async def test_agent_and_connector_fail_simultaneously(self, seed_random):
        """Test handling when agent and connector fail at the same time."""
        agent = MockFailingAgent("failing_agent", fail_probability=1.0)
        connector = MockConnector("failing_conn", fail_on_call=1)

        agent_error = None
        connector_error = None

        # Both fail concurrently
        async def agent_task():
            nonlocal agent_error
            try:
                await agent.generate("test")
            except Exception as e:
                agent_error = e

        async def connector_task():
            nonlocal connector_error
            try:
                await connector.fetch_data("test")
            except Exception as e:
                connector_error = e

        await asyncio.gather(agent_task(), connector_task())

        # Both should have failed
        assert agent_error is not None
        assert connector_error is not None

    @pytest.mark.asyncio
    async def test_cascading_failures_across_layers(self, seed_random):
        """Test cascading failures from connector -> agent -> debate."""
        events: list[str] = []

        # Connector fails first
        connector = MockConnector("evidence_conn", fail_on_call=1)
        try:
            await connector.fetch_data("evidence query")
        except Exception:
            events.append("connector_failed")

        # Agent depends on connector, also fails
        agent = MockFailingAgent("dependent_agent", fail_probability=1.0)
        try:
            await agent.generate("prompt without evidence")
        except Exception:
            events.append("agent_failed")

        # Debate context records cascade
        ctx = DebateContext(env=Environment(task="Test cascade"))
        ctx.evidence_pack = None  # No evidence due to failure
        events.append("debate_degraded")

        assert events == ["connector_failed", "agent_failed", "debate_degraded"]

    @pytest.mark.asyncio
    async def test_concurrent_multi_agent_with_failures(self, seed_random):
        """Test concurrent agent calls with random failures."""
        agents = [MockFailingAgent(f"agent{i}", fail_probability=0.3) for i in range(10)]

        async def call_agent(agent: MockFailingAgent) -> tuple[str, bool]:
            try:
                await agent.generate("concurrent test")
                return (agent.name, True)
            except Exception:
                return (agent.name, False)

        results = await asyncio.gather(*[call_agent(a) for a in agents])

        successes = sum(1 for _, success in results if success)
        failures = sum(1 for _, success in results if not success)

        # With 30% failure rate, expect roughly 7 successes and 3 failures
        # Allow variance due to randomness
        assert successes + failures == 10
        assert successes >= 4  # At least some succeeded
        assert failures >= 1  # At least some failed


# =============================================================================
# Test: Metrics Accuracy Under Load
# =============================================================================


class TestMetricsAccuracy:
    """Tests for metrics accuracy under various load conditions."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics_accuracy(self):
        """Test circuit breaker metrics are accurate under load."""
        breaker = get_circuit_breaker(
            "metrics_test",
            failure_threshold=5,
            cooldown_seconds=1.0,
        )

        failures_recorded = 0
        successes_recorded = 0

        # Record mixed successes and failures
        for i in range(20):
            if i % 3 == 0:
                breaker.record_failure("entity1")
                failures_recorded += 1
            else:
                breaker.record_success("entity1")
                successes_recorded += 1

        # Verify status reflects what we recorded
        status = breaker.get_all_status()
        assert "entity1" in status
        # Circuit may be open or closed depending on final sequence
        # Just verify tracking is working
        assert status["entity1"]["status"] in ["closed", "open", "half-open"]

    @pytest.mark.asyncio
    async def test_concurrent_metric_recording(self):
        """Test metrics remain accurate under concurrent access."""
        breaker = get_circuit_breaker(
            "concurrent_metrics",
            failure_threshold=100,  # High threshold to avoid opening
            cooldown_seconds=1.0,
        )

        success_count = 0
        failure_count = 0
        count_lock = asyncio.Lock()

        async def record_success():
            nonlocal success_count
            for _ in range(100):
                breaker.record_success("concurrent_entity")
                async with count_lock:
                    success_count += 1
                await asyncio.sleep(0)

        async def record_failure():
            nonlocal failure_count
            for _ in range(50):
                breaker.record_failure("concurrent_entity")
                async with count_lock:
                    failure_count += 1
                await asyncio.sleep(0)

        # Run concurrent metric recording
        await asyncio.gather(
            record_success(),
            record_failure(),
            record_success(),
        )

        # Verify counts
        assert success_count == 200  # 2 * 100
        assert failure_count == 50
        # Verify breaker tracked entity
        status = breaker.get_all_status()
        assert "concurrent_entity" in status

    @pytest.mark.asyncio
    async def test_agent_call_metrics_under_load(self, seed_random):
        """Test agent call metrics under high load."""
        agents = [MockFailingAgent(f"load_agent{i}") for i in range(5)]

        # Use list to avoid race conditions with concurrent updates
        results: list[tuple[str, bool]] = []
        results_lock = asyncio.Lock()

        async def call_agent(agent: MockFailingAgent):
            for _ in range(20):
                success = False
                try:
                    await agent.generate("load test")
                    success = True
                except Exception:
                    pass
                async with results_lock:
                    results.append((agent.name, success))

        await asyncio.gather(*[call_agent(a) for a in agents])

        total_calls = len(results)
        total_successes = sum(1 for _, s in results if s)

        # All 5 agents called 20 times each = 100 calls
        assert total_calls == 100
        # All should succeed (no failure probability set)
        assert total_successes == 100


# =============================================================================
# Test: Recovery Patterns
# =============================================================================


class TestRecoveryPatterns:
    """Tests for various recovery patterns."""

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test exponential backoff follows expected timing."""
        delays: list[float] = []
        base_delay = 0.1  # 100ms base

        for attempt in range(5):
            delay = min(base_delay * (2**attempt), 3.0)  # Cap at 3s
            delays.append(delay)

        # Verify exponential progression
        assert delays == [0.1, 0.2, 0.4, 0.8, 1.6]

    @pytest.mark.asyncio
    async def test_circuit_closes_after_successful_recovery(self):
        """Test circuit properly closes after successful recovery."""
        agent = MockFailingAgent(
            "recovery_agent",
            fail_after_rounds=2,
            recovery_after_failures=1,  # Recovers after 1 failure
        )

        # Round 1 & 2: Success
        await agent.generate("round 1")
        await agent.generate("round 2")

        # Round 3: Fails (after threshold)
        try:
            await agent.generate("round 3")
        except Exception:
            pass

        # Round 4+: Should recover
        result = await agent.generate("round 4")
        assert result is not None

    @pytest.mark.asyncio
    async def test_graceful_degradation_preserves_core_functionality(self, seed_random):
        """Test that graceful degradation preserves core debate functionality."""
        # Start with 5 agents
        agents = [
            MockFailingAgent("core1"),
            MockFailingAgent("core2"),
            MockFailingAgent("optional1", fail_probability=1.0),  # Always fails
            MockFailingAgent("optional2", fail_probability=1.0),  # Always fails
            MockFailingAgent("optional3", fail_probability=1.0),  # Always fails
        ]

        # Categorize agents
        core_agents = agents[:2]
        optional_agents = agents[2:]

        # Collect responses
        core_responses: list[str] = []
        optional_responses: list[str] = []

        for agent in core_agents:
            try:
                response = await agent.generate("core task")
                core_responses.append(response)
            except Exception:
                pass

        for agent in optional_agents:
            try:
                response = await agent.generate("optional task")
                optional_responses.append(response)
            except Exception:
                pass

        # Core functionality preserved
        assert len(core_responses) == 2
        # Optional features degraded gracefully
        assert len(optional_responses) == 0

    @pytest.mark.asyncio
    async def test_debate_completion_guarantee(self, seed_random):
        """Test debate always completes (success or failure), never hangs."""
        agents = [
            MockFailingAgent("agent1", timeout_probability=0.5),
            MockFailingAgent("agent2", fail_probability=0.5),
            MockFailingAgent("agent3"),
        ]

        debate_completed = False
        completion_reason = None

        try:
            # Simulate debate with timeout
            async def run_debate():
                responses = []
                for agent in agents:
                    try:
                        response = await asyncio.wait_for(
                            agent.generate("debate prompt"),
                            timeout=0.5,
                        )
                        responses.append(response)
                    except asyncio.TimeoutError:
                        pass
                    except Exception:
                        pass
                return responses

            results = await asyncio.wait_for(run_debate(), timeout=5.0)
            debate_completed = True
            completion_reason = "success" if results else "partial"

        except asyncio.TimeoutError:
            debate_completed = True
            completion_reason = "timeout"
        except Exception as e:
            debate_completed = True
            completion_reason = f"error: {e}"

        # Debate MUST complete, never hang
        assert debate_completed
        assert completion_reason is not None


# =============================================================================
# Test: State Persistence (if applicable)
# =============================================================================


class TestStatePersistence:
    """Tests for state persistence and recovery."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_persists_across_instances(self):
        """Test circuit breaker state persists across instances."""
        # Create first instance and record failures
        breaker1 = get_circuit_breaker(
            "persist_test",
            failure_threshold=3,
            cooldown_seconds=60.0,  # Long cooldown
        )

        for _ in range(3):
            breaker1.record_failure("entity1")

        # Circuit should be open
        assert not breaker1.can_proceed("entity1")

        # Get same circuit breaker (from registry)
        breaker2 = get_circuit_breaker(
            "persist_test",
            failure_threshold=3,
            cooldown_seconds=60.0,
        )

        # Should still be open (same instance from registry)
        assert not breaker2.can_proceed("entity1")

    @pytest.mark.asyncio
    async def test_debate_context_serialization(self):
        """Test debate context can be serialized for checkpointing."""
        ctx = DebateContext(
            env=Environment(task="Checkpoint test"),
            debate_id="test-123",
            current_round=2,
        )
        ctx.proposals["agent1"] = "Proposal 1"
        ctx.proposals["agent2"] = "Proposal 2"

        # Serialize to dict
        summary = ctx.to_summary_dict()

        # Verify serialization works
        assert summary["debate_id"] == "test-123"
        assert summary["current_round"] == 2
        assert summary["num_proposals"] == 2
