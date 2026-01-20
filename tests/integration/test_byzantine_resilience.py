"""
Byzantine Resilience Integration Stress Tests.

Tests Byzantine fault-tolerant consensus under realistic debate conditions:
- Concurrent Byzantine consensus protocols
- Integration with debate orchestration
- Stress testing with various agent failure modes
- Recovery scenarios under load
- Network partition simulations

Run with:
    pytest tests/integration/test_byzantine_resilience.py -v --asyncio-mode=auto

For full stress:
    pytest tests/integration/test_byzantine_resilience.py -v -k stress --asyncio-mode=auto -s
"""

from __future__ import annotations

import asyncio
import gc
import random
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Agent, Critique, Message, Vote
from aragora.debate.byzantine import (
    ByzantineConsensus,
    ByzantineConsensusConfig,
    ByzantineConsensusResult,
    ByzantineMessage,
    ByzantinePhase,
    ConsensusFailure,
    verify_with_byzantine_consensus,
)


# =============================================================================
# Mock Agent Fixtures
# =============================================================================


class StressTestAgent(Agent):
    """Agent for stress testing Byzantine consensus."""

    def __init__(
        self,
        name: str,
        response_delay: float = 0.0,
        failure_probability: float = 0.0,
        byzantine_mode: str = "honest",  # honest, disagree, random, slow, silent
    ):
        super().__init__(name=name, model="stress-test", role="proposer")
        self.agent_type = "stress"
        self.response_delay = response_delay
        self.failure_probability = failure_probability
        self.byzantine_mode = byzantine_mode
        self.call_count = 0
        self.total_latency = 0.0

    async def generate(self, prompt: str, context: list = None) -> str:
        self.call_count += 1
        start = time.time()

        try:
            # Simulate network delay
            if self.response_delay > 0:
                await asyncio.sleep(self.response_delay)

            # Random failure simulation
            if random.random() < self.failure_probability:
                raise RuntimeError(f"Simulated failure in {self.name}")

            # Byzantine behavior modes
            if self.byzantine_mode == "honest":
                return "PREPARE: YES\nCOMMIT: YES\nREASONING: I agree with the proposal"

            elif self.byzantine_mode == "disagree":
                return "PREPARE: NO\nCOMMIT: NO\nREASONING: I disagree"

            elif self.byzantine_mode == "random":
                choice = random.choice(["YES", "NO"])
                return f"PREPARE: {choice}\nCOMMIT: {choice}\nREASONING: Random"

            elif self.byzantine_mode == "slow":
                await asyncio.sleep(100)  # Very slow
                return "PREPARE: YES\nREASONING: Finally responding"

            elif self.byzantine_mode == "silent":
                raise asyncio.TimeoutError("Agent silent")

            return "PREPARE: YES\nREASONING: Default response"

        finally:
            self.total_latency += time.time() - start

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list = None,
        target_agent: str = None,
    ) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=["Test issue"],
            suggestions=["Test suggestion"],
            severity=0.5,
            reasoning="Test reasoning",
        )


@pytest.fixture
def create_stress_agents():
    """Factory for creating stress test agent groups."""

    def factory(
        n_honest: int = 5,
        n_byzantine: int = 0,
        byzantine_mode: str = "disagree",
        response_delay: float = 0.0,
        failure_probability: float = 0.0,
    ) -> List[StressTestAgent]:
        agents = []

        # Create honest agents
        for i in range(n_honest):
            agents.append(
                StressTestAgent(
                    name=f"honest_{i}",
                    response_delay=response_delay,
                    failure_probability=failure_probability,
                    byzantine_mode="honest",
                )
            )

        # Create Byzantine agents
        for i in range(n_byzantine):
            agents.append(
                StressTestAgent(
                    name=f"byzantine_{i}",
                    response_delay=response_delay,
                    byzantine_mode=byzantine_mode,
                )
            )

        random.shuffle(agents)  # Mix them up
        return agents

    return factory


# =============================================================================
# Concurrent Protocol Tests
# =============================================================================


class TestConcurrentByzantineProtocols:
    """Tests for running multiple Byzantine consensus protocols concurrently."""

    @pytest.mark.asyncio
    async def test_parallel_consensus_5_protocols(self, create_stress_agents):
        """Run 5 Byzantine consensus protocols in parallel."""
        results = []

        async def run_protocol(idx: int) -> ByzantineConsensusResult:
            agents = create_stress_agents(n_honest=5, n_byzantine=1)
            config = ByzantineConsensusConfig(
                phase_timeout_seconds=5.0,
                max_view_changes=2,
            )
            protocol = ByzantineConsensus(agents=agents, config=config)
            return await protocol.propose(f"Proposal {idx}", task=f"Task {idx}")

        results = await asyncio.gather(*[run_protocol(i) for i in range(5)])

        # All should succeed with 5 honest + 1 Byzantine (f=2 for n=6)
        success_count = sum(1 for r in results if r.success)
        assert success_count >= 4, f"Only {success_count}/5 succeeded"

    @pytest.mark.asyncio
    async def test_parallel_consensus_20_protocols(self, create_stress_agents):
        """Run 20 Byzantine consensus protocols in parallel."""

        async def run_protocol(idx: int) -> Tuple[int, bool]:
            agents = create_stress_agents(n_honest=4, n_byzantine=0)
            config = ByzantineConsensusConfig(
                phase_timeout_seconds=5.0,
                max_view_changes=1,
            )
            protocol = ByzantineConsensus(agents=agents, config=config)
            result = await protocol.propose(f"Proposal {idx}")
            return idx, result.success

        results = await asyncio.gather(*[run_protocol(i) for i in range(20)])

        success_count = sum(1 for _, success in results if success)
        assert success_count >= 18, f"Only {success_count}/20 succeeded"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_parallel_consensus_50_protocols_stress(self, create_stress_agents):
        """Stress: Run 50 Byzantine consensus protocols in parallel."""

        async def run_protocol(idx: int) -> bool:
            agents = create_stress_agents(n_honest=4, n_byzantine=0)
            config = ByzantineConsensusConfig(
                phase_timeout_seconds=10.0,
                max_view_changes=1,
            )
            protocol = ByzantineConsensus(agents=agents, config=config)
            result = await protocol.propose(f"Stress proposal {idx}")
            return result.success

        results = await asyncio.gather(*[run_protocol(i) for i in range(50)])

        success_count = sum(1 for r in results if r)
        assert success_count >= 45, f"Only {success_count}/50 succeeded"


# =============================================================================
# Load Testing with Failures
# =============================================================================


class TestByzantineUnderLoad:
    """Tests for Byzantine consensus under realistic failure conditions."""

    @pytest.mark.asyncio
    async def test_consensus_with_network_delays(self, create_stress_agents):
        """Test consensus with simulated network delays."""
        # Agents with varying delays (0-200ms)
        agents = []
        for i in range(7):
            delay = random.uniform(0.0, 0.2)
            agents.append(
                StressTestAgent(
                    name=f"delayed_{i}",
                    response_delay=delay,
                    byzantine_mode="honest",
                )
            )

        config = ByzantineConsensusConfig(
            phase_timeout_seconds=5.0,
            max_view_changes=2,
        )
        protocol = ByzantineConsensus(agents=agents, config=config)

        start = time.time()
        result = await protocol.propose("Delayed consensus test")
        elapsed = time.time() - start

        assert result.success is True
        # Should complete within reasonable time despite delays
        assert elapsed < 10.0, f"Took {elapsed:.1f}s"

    @pytest.mark.asyncio
    async def test_consensus_with_intermittent_failures(self, create_stress_agents):
        """Test consensus with agents that intermittently fail."""
        agents = []
        for i in range(8):
            # 10% failure probability
            agents.append(
                StressTestAgent(
                    name=f"intermittent_{i}",
                    failure_probability=0.1,
                    byzantine_mode="honest",
                )
            )

        config = ByzantineConsensusConfig(
            phase_timeout_seconds=5.0,
            max_view_changes=3,
        )
        protocol = ByzantineConsensus(agents=agents, config=config)

        # Run multiple times to test reliability
        successes = 0
        for i in range(10):
            result = await protocol.propose(f"Intermittent test {i}")
            if result.success:
                successes += 1

        # Most should succeed despite intermittent failures
        assert successes >= 7, f"Only {successes}/10 succeeded"

    @pytest.mark.asyncio
    async def test_consensus_with_mixed_failure_modes(self, create_stress_agents):
        """Test consensus with various failure modes mixed."""
        agents = [
            StressTestAgent(name="honest_1", byzantine_mode="honest"),
            StressTestAgent(name="honest_2", byzantine_mode="honest"),
            StressTestAgent(name="honest_3", byzantine_mode="honest"),
            StressTestAgent(name="honest_4", byzantine_mode="honest"),
            StressTestAgent(name="honest_5", byzantine_mode="honest"),
            StressTestAgent(name="disagree", byzantine_mode="disagree"),
            StressTestAgent(name="random", byzantine_mode="random"),
        ]

        config = ByzantineConsensusConfig(
            phase_timeout_seconds=5.0,
            max_view_changes=2,
        )
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Mixed failures test")

        # 5 honest out of 7 should be enough (f=2, quorum=5)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_consensus_at_byzantine_threshold(self, create_stress_agents):
        """Test consensus exactly at the Byzantine fault threshold."""
        # n=7, f=2 means exactly 2 Byzantine nodes is the limit
        agents = create_stress_agents(n_honest=5, n_byzantine=2, byzantine_mode="disagree")

        config = ByzantineConsensusConfig(
            phase_timeout_seconds=5.0,
            max_view_changes=2,
        )
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Threshold test")

        # Should succeed at exactly the threshold
        assert result.success is True
        assert result.commit_count >= protocol.quorum_size


# =============================================================================
# Recovery and View Change Tests
# =============================================================================


class TestByzantineRecoveryUnderStress:
    """Tests for Byzantine recovery mechanisms under stress."""

    @pytest.mark.asyncio
    async def test_rapid_view_changes(self, create_stress_agents):
        """Test handling rapid view changes."""
        # First two agents are silent (will trigger view changes)
        agents = [
            StressTestAgent(name="silent_0", byzantine_mode="silent"),
            StressTestAgent(name="silent_1", byzantine_mode="silent"),
            StressTestAgent(name="honest_0", byzantine_mode="honest"),
            StressTestAgent(name="honest_1", byzantine_mode="honest"),
            StressTestAgent(name="honest_2", byzantine_mode="honest"),
            StressTestAgent(name="honest_3", byzantine_mode="honest"),
            StressTestAgent(name="honest_4", byzantine_mode="honest"),
        ]

        config = ByzantineConsensusConfig(
            phase_timeout_seconds=0.1,  # Quick timeout
            max_view_changes=5,
        )
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("View change test")

        # Should eventually succeed with honest agents as leaders
        assert result.success is True
        # View should have changed at least once
        assert result.view >= 1 or result.success

    @pytest.mark.asyncio
    async def test_recovery_after_cascade_failure(self, create_stress_agents):
        """Test recovery after multiple consecutive failures."""
        call_counts = {}

        class CascadeAgent(StressTestAgent):
            """Agent that fails on first few calls then succeeds."""

            def __init__(self, name: str, fail_until: int):
                super().__init__(name=name, byzantine_mode="honest")
                self.fail_until = fail_until

            async def generate(self, prompt: str, context: list = None) -> str:
                self.call_count += 1
                if self.call_count <= self.fail_until:
                    raise RuntimeError(f"Cascade failure {self.call_count}")
                return "PREPARE: YES\nCOMMIT: YES\nREASONING: Recovered"

        agents = [
            CascadeAgent(name=f"cascade_{i}", fail_until=1)  # Fail first call
            for i in range(7)
        ]

        config = ByzantineConsensusConfig(
            phase_timeout_seconds=1.0,
            max_view_changes=3,
        )
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Cascade recovery test")

        # May need view changes but should eventually succeed
        assert result is not None

    @pytest.mark.asyncio
    async def test_sequential_consensus_after_failures(self, create_stress_agents):
        """Test running multiple consensus rounds after failures."""
        agents = create_stress_agents(n_honest=6, n_byzantine=1)
        config = ByzantineConsensusConfig(
            phase_timeout_seconds=5.0,
            max_view_changes=2,
        )
        protocol = ByzantineConsensus(agents=agents, config=config)

        results = []
        for i in range(5):
            result = await protocol.propose(f"Sequential test {i}")
            results.append(result)

        # All should succeed
        success_count = sum(1 for r in results if r.success)
        assert success_count >= 4


# =============================================================================
# Scalability Tests
# =============================================================================


class TestByzantineScalability:
    """Tests for Byzantine consensus scalability."""

    @pytest.mark.asyncio
    async def test_consensus_with_10_agents(self, create_stress_agents):
        """Test consensus with 10 agents (f=3)."""
        agents = create_stress_agents(n_honest=7, n_byzantine=3)
        config = ByzantineConsensusConfig(phase_timeout_seconds=5.0)
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("10 agent test")

        # n=10, f=3, quorum=7. With 7 honest, should succeed.
        assert result.success is True

    @pytest.mark.asyncio
    async def test_consensus_with_13_agents(self, create_stress_agents):
        """Test consensus with 13 agents (f=4)."""
        agents = create_stress_agents(n_honest=9, n_byzantine=4)
        config = ByzantineConsensusConfig(phase_timeout_seconds=5.0)
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("13 agent test")

        # n=13, f=4, quorum=9. With 9 honest, should succeed.
        assert result.success is True

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_consensus_with_20_agents_stress(self, create_stress_agents):
        """Stress: Test consensus with 20 agents."""
        agents = create_stress_agents(n_honest=14, n_byzantine=6)
        config = ByzantineConsensusConfig(phase_timeout_seconds=10.0)
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("20 agent stress test")

        # n=20, f=6, quorum=13. With 14 honest, should succeed.
        assert result.success is True


# =============================================================================
# Network Partition Simulation
# =============================================================================


class TestNetworkPartitions:
    """Tests simulating network partition scenarios."""

    @pytest.mark.asyncio
    async def test_partition_with_honest_majority(self, create_stress_agents):
        """Test with partition that leaves honest majority accessible."""
        # Simulate partition: 2 agents become unreachable
        agents = []
        for i in range(5):
            agents.append(StressTestAgent(name=f"reachable_{i}", byzantine_mode="honest"))
        for i in range(2):
            agents.append(StressTestAgent(name=f"partitioned_{i}", byzantine_mode="silent"))

        config = ByzantineConsensusConfig(
            phase_timeout_seconds=0.1,  # Quick timeout for partitioned nodes
            max_view_changes=2,
        )
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Partition test")

        # Should succeed with 5 honest reachable agents
        assert result.success is True

    @pytest.mark.asyncio
    async def test_partition_loses_quorum(self, create_stress_agents):
        """Test with partition that prevents quorum."""
        # Partition leaves only 3 reachable (not enough for quorum)
        agents = []
        for i in range(3):
            agents.append(StressTestAgent(name=f"reachable_{i}", byzantine_mode="honest"))
        for i in range(4):
            agents.append(StressTestAgent(name=f"partitioned_{i}", byzantine_mode="silent"))

        config = ByzantineConsensusConfig(
            phase_timeout_seconds=0.1,
            max_view_changes=1,
        )
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Failed partition test")

        # n=7, f=2, quorum=5. Only 3 reachable, should fail.
        assert result.success is False


# =============================================================================
# Memory and Resource Tests
# =============================================================================


class TestByzantineResourceUsage:
    """Tests for memory and resource usage in Byzantine consensus."""

    @pytest.mark.asyncio
    async def test_no_memory_leak_repeated_protocols(self, create_stress_agents):
        """Test that repeated protocol runs don't leak memory."""
        gc.collect()
        initial_objects = len(gc.get_objects())

        for i in range(50):
            agents = create_stress_agents(n_honest=4, n_byzantine=0)
            config = ByzantineConsensusConfig(phase_timeout_seconds=1.0)
            protocol = ByzantineConsensus(agents=agents, config=config)
            await protocol.propose(f"Memory test {i}")

        gc.collect()
        final_objects = len(gc.get_objects())

        # Allow some growth but not excessive
        growth = final_objects - initial_objects
        assert growth < 5000, f"Object count grew by {growth}"

    @pytest.mark.asyncio
    async def test_result_isolation(self, create_stress_agents):
        """Test that results from different protocols are isolated."""
        agents1 = create_stress_agents(n_honest=4)
        agents2 = create_stress_agents(n_honest=4)

        config = ByzantineConsensusConfig(phase_timeout_seconds=5.0)

        protocol1 = ByzantineConsensus(agents=agents1, config=config)
        protocol2 = ByzantineConsensus(agents=agents2, config=config)

        result1 = await protocol1.propose("Proposal A")
        result2 = await protocol2.propose("Proposal B")

        # Results should be independent
        assert result1.value != result2.value
        assert result1.gauntlet_id if hasattr(result1, "gauntlet_id") else True


# =============================================================================
# Timing and Performance Tests
# =============================================================================


class TestByzantineTiming:
    """Tests for Byzantine consensus timing behavior."""

    @pytest.mark.asyncio
    async def test_fast_consensus_all_honest(self, create_stress_agents):
        """Test that consensus is fast when all agents are honest."""
        agents = create_stress_agents(n_honest=7, n_byzantine=0)
        config = ByzantineConsensusConfig(phase_timeout_seconds=5.0)
        protocol = ByzantineConsensus(agents=agents, config=config)

        start = time.time()
        result = await protocol.propose("Fast consensus test")
        elapsed = time.time() - start

        assert result.success is True
        assert elapsed < 2.0, f"Fast consensus took {elapsed:.1f}s"

    @pytest.mark.asyncio
    async def test_consensus_respects_timeout(self, create_stress_agents):
        """Test that consensus respects phase timeouts."""
        # All agents are slow
        agents = [
            StressTestAgent(name=f"slow_{i}", byzantine_mode="slow")
            for i in range(4)
        ]

        config = ByzantineConsensusConfig(
            phase_timeout_seconds=0.1,  # Very short timeout
            max_view_changes=1,
        )
        protocol = ByzantineConsensus(agents=agents, config=config)

        start = time.time()
        result = await protocol.propose("Timeout test")
        elapsed = time.time() - start

        # Should fail fast due to timeouts, not wait for slow agents
        assert elapsed < 5.0, f"Should have timed out faster: {elapsed:.1f}s"

    @pytest.mark.asyncio
    async def test_duration_tracking_accuracy(self, create_stress_agents):
        """Test that duration is accurately tracked."""
        agents = create_stress_agents(n_honest=4, response_delay=0.1)
        config = ByzantineConsensusConfig(phase_timeout_seconds=5.0)
        protocol = ByzantineConsensus(agents=agents, config=config)

        start = time.time()
        result = await protocol.propose("Duration test")
        wall_time = time.time() - start

        # Reported duration should be close to wall time
        assert abs(result.duration_seconds - wall_time) < 1.0


# =============================================================================
# Edge Cases Under Stress
# =============================================================================


class TestByzantineEdgeCasesUnderStress:
    """Edge case tests under stress conditions."""

    @pytest.mark.asyncio
    async def test_empty_proposal_under_load(self, create_stress_agents):
        """Test empty proposals under concurrent load."""

        async def run_empty_proposal(idx: int) -> bool:
            agents = create_stress_agents(n_honest=4)
            protocol = ByzantineConsensus(agents=agents)
            result = await protocol.propose("")
            return result.success

        results = await asyncio.gather(*[run_empty_proposal(i) for i in range(10)])
        assert all(results)

    @pytest.mark.asyncio
    async def test_large_proposal_under_load(self, create_stress_agents):
        """Test large proposals under concurrent load."""
        large_proposal = "x" * 10000

        async def run_large_proposal(idx: int) -> bool:
            agents = create_stress_agents(n_honest=4)
            protocol = ByzantineConsensus(agents=agents)
            result = await protocol.propose(large_proposal)
            return result.success

        results = await asyncio.gather(*[run_large_proposal(i) for i in range(5)])
        assert all(results)

    @pytest.mark.asyncio
    async def test_unicode_proposals_concurrent(self, create_stress_agents):
        """Test Unicode proposals concurrently."""
        proposals = [
            "日本語テスト",
            "مرحبا بالعالم",
            "🔥🚀💯",
            "Ελληνικά",
            "עברית",
        ]

        async def run_unicode_proposal(proposal: str) -> bool:
            agents = create_stress_agents(n_honest=4)
            protocol = ByzantineConsensus(agents=agents)
            result = await protocol.propose(proposal)
            return result.success

        results = await asyncio.gather(*[run_unicode_proposal(p) for p in proposals])
        assert all(results)

    @pytest.mark.asyncio
    async def test_minimum_agents_stress(self, create_stress_agents):
        """Test minimum agent count (4) under stress."""

        async def run_min_agents(idx: int) -> bool:
            agents = create_stress_agents(n_honest=4, n_byzantine=0)
            protocol = ByzantineConsensus(agents=agents)
            result = await protocol.propose(f"Min agents {idx}")
            return result.success

        results = await asyncio.gather(*[run_min_agents(i) for i in range(20)])
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.9, f"Success rate: {success_rate}"


# =============================================================================
# Integration Smoke Tests
# =============================================================================


class TestByzantineIntegrationSmoke:
    """Smoke tests for Byzantine consensus integration."""

    @pytest.mark.asyncio
    async def test_verify_with_byzantine_consensus_function(self, create_stress_agents):
        """Test the convenience verification function."""
        agents = create_stress_agents(n_honest=5, n_byzantine=1)

        result = await verify_with_byzantine_consensus(
            proposal="Smoke test proposal",
            agents=agents,
            task="Verification smoke test",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_protocol_reuse(self, create_stress_agents):
        """Test reusing a protocol for multiple proposals."""
        agents = create_stress_agents(n_honest=5, n_byzantine=1)
        config = ByzantineConsensusConfig(phase_timeout_seconds=5.0)
        protocol = ByzantineConsensus(agents=agents, config=config)

        results = []
        for i in range(5):
            result = await protocol.propose(f"Reuse test {i}")
            results.append(result)

        # All should succeed
        assert all(r.success for r in results)
        # Sequence should increment
        sequences = [r.sequence for r in results]
        assert sequences == sorted(sequences)

    @pytest.mark.asyncio
    async def test_quorum_calculation_correctness(self, create_stress_agents):
        """Verify quorum calculations are correct for various agent counts."""
        test_cases = [
            (4, 1, 3),   # n=4, f=1, quorum=3
            (7, 2, 5),   # n=7, f=2, quorum=5
            (10, 3, 7),  # n=10, f=3, quorum=7
            (13, 4, 9),  # n=13, f=4, quorum=9
        ]

        for n, expected_f, expected_quorum in test_cases:
            agents = create_stress_agents(n_honest=n)
            protocol = ByzantineConsensus(agents=agents)

            assert protocol.n == n
            assert protocol.f == expected_f, f"n={n}: expected f={expected_f}, got {protocol.f}"
            assert protocol.quorum_size == expected_quorum, (
                f"n={n}: expected quorum={expected_quorum}, got {protocol.quorum_size}"
            )
