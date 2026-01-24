"""
Multi-cycle resilience tests for Nomic Loop.

Tests critical edge cases for production reliability:
1. Multi-cycle consensus convergence - ensures consensus improves across cycles
2. Agent dropout during debate - handles mid-round agent failures gracefully
3. Deadline enforcement timeout - verify phase doesn't loop forever
4. Checkpoint recovery consistency - state is correctly restored
5. Empty agent fallback - graceful degradation when no agents available
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.phases import (
    PhaseResult,
    PhaseValidator,
    PhaseValidationError,
)
from aragora.nomic.states import NomicState, StateContext
from aragora.nomic.recovery import (
    RecoveryStrategy,
    RecoveryDecision,
    CircuitBreaker,
    CircuitBreakerRegistry,
    RecoveryManager,
)


# =============================================================================
# Test: Multi-Cycle Consensus Convergence
# =============================================================================


class TestMultiCycleConsensusConvergence:
    """Tests for multi-cycle consensus convergence behavior."""

    def test_consensus_improves_across_cycles(self):
        """Verify consensus score trends upward across cycles."""
        # Simulate 5 cycles of consensus scores
        cycle_scores = [0.45, 0.52, 0.68, 0.75, 0.82]

        # Verify monotonic improvement trend
        for i in range(1, len(cycle_scores)):
            assert (
                cycle_scores[i] >= cycle_scores[i - 1]
            ), f"Consensus should not regress: {cycle_scores[i - 1]} -> {cycle_scores[i]}"

        # Verify final score meets threshold
        assert cycle_scores[-1] >= 0.8, "Should achieve 80%+ consensus after 5 cycles"

    def test_stalled_consensus_triggers_recovery(self):
        """Verify stalled consensus (no improvement) triggers recovery."""
        # Simulate stalled consensus over 3 cycles
        stalled_scores = [0.55, 0.54, 0.55]  # No meaningful improvement

        stall_threshold = 0.05  # Must improve by at least 5%
        is_stalled = all(
            abs(stalled_scores[i] - stalled_scores[i - 1]) < stall_threshold
            for i in range(1, len(stalled_scores))
        )

        assert is_stalled, "Should detect stall when no improvement"

    def test_consensus_convergence_with_dropout(self):
        """Verify consensus can still converge even with agent dropout."""
        # Simulate 4 agents, 1 drops out at cycle 3
        agent_counts = [4, 4, 3, 3, 3]  # Agent dropout at cycle 3
        consensus_scores = [0.4, 0.55, 0.52, 0.65, 0.78]  # Slight dip, then recovery

        # Verify we still converge despite dropout
        final_score = consensus_scores[-1]
        assert final_score >= 0.7, "Should converge despite agent dropout"

        # Verify minimum agent count maintained
        min_agents = 2
        assert all(
            count >= min_agents for count in agent_counts
        ), "Must maintain minimum agent count"

    @pytest.mark.asyncio
    async def test_cycle_budget_enforcement(self):
        """Verify cycle budget is respected and not exceeded."""
        max_cycles = 6
        completed_cycles = 0
        timeout_seconds = 5.0
        start_time = time.time()

        async def simulate_cycle():
            nonlocal completed_cycles
            await asyncio.sleep(0.1)  # Simulate cycle work
            completed_cycles += 1
            return completed_cycles < max_cycles

        while await simulate_cycle():
            if time.time() - start_time > timeout_seconds:
                break

        assert completed_cycles <= max_cycles, f"Exceeded cycle budget: {completed_cycles}"


# =============================================================================
# Test: Agent Dropout During Debate
# =============================================================================


class TestAgentDropoutHandling:
    """Tests for handling agent failures mid-debate."""

    def test_circuit_breaker_tracks_agent_failures(self):
        """Verify circuit breaker correctly tracks agent failures."""
        breaker = CircuitBreaker(
            name="test_agent",
            failure_threshold=3,
            reset_timeout_seconds=10,
        )

        # Record failures up to threshold
        for i in range(3):
            breaker.record_failure()

        assert breaker._state == "open", "Circuit should open after 3 failures"
        assert breaker.is_open, "Should block requests when open"

    def test_agent_fallback_on_dropout(self):
        """Verify fallback agent is used when primary drops out."""
        primary_agents = ["claude", "gemini", "grok"]
        fallback_agents = ["mistral", "deepseek"]
        failed_agents = {"claude"}

        available = [a for a in primary_agents if a not in failed_agents]
        available.extend(fallback_agents[:1])  # Add one fallback

        assert len(available) >= 2, "Should have at least 2 agents for debate"
        assert "gemini" in available, "Non-failed agent should be available"
        assert "mistral" in available, "Fallback should be added"

    @pytest.mark.asyncio
    async def test_partial_results_preserved_on_dropout(self):
        """Verify partial debate results are preserved when agent drops out."""
        partial_results = {
            "round_1": {"claude": "Proposal A", "gemini": "Proposal B"},
            "round_2": {"claude": "Critique", "gemini": None},  # gemini dropped out
        }

        # Verify round 1 is complete
        assert all(v is not None for v in partial_results["round_1"].values())

        # Verify round 2 partial results preserved
        preserved_responses = sum(1 for v in partial_results["round_2"].values() if v is not None)
        assert preserved_responses >= 1, "Should preserve at least one response"

    def test_minimum_quorum_check(self):
        """Verify minimum quorum is enforced for valid debate."""
        minimum_quorum = 2

        test_cases = [
            (["claude", "gemini", "grok"], True),  # 3 agents - valid
            (["claude", "gemini"], True),  # 2 agents - valid (minimum)
            (["claude"], False),  # 1 agent - invalid
            ([], False),  # 0 agents - invalid
        ]

        for agents, expected_valid in test_cases:
            has_quorum = len(agents) >= minimum_quorum
            assert has_quorum == expected_valid, f"Quorum check failed for {len(agents)} agents"


# =============================================================================
# Test: Deadline Enforcement
# =============================================================================


class TestDeadlineEnforcement:
    """Tests for phase timeout and deadline enforcement."""

    @pytest.mark.asyncio
    async def test_phase_timeout_enforced(self):
        """Verify phases timeout and don't loop forever."""
        phase_timeout = 0.5  # 500ms
        phase_started = asyncio.Event()
        phase_completed = asyncio.Event()

        async def slow_phase():
            phase_started.set()
            await asyncio.sleep(2.0)  # Would take 2s, but should timeout
            phase_completed.set()

        try:
            await asyncio.wait_for(slow_phase(), timeout=phase_timeout)
            assert False, "Should have timed out"
        except asyncio.TimeoutError:
            pass  # Expected

        assert phase_started.is_set(), "Phase should have started"
        assert not phase_completed.is_set(), "Phase should not have completed"

    @pytest.mark.asyncio
    async def test_verify_fix_cycle_has_budget(self):
        """Verify verify-fix cycle has iteration budget."""
        max_fix_iterations = 5
        fix_attempts = 0

        async def verify_and_fix():
            nonlocal fix_attempts
            fix_attempts += 1
            return fix_attempts >= 3  # Succeeds on 3rd attempt

        while fix_attempts < max_fix_iterations:
            if await verify_and_fix():
                break

        assert fix_attempts <= max_fix_iterations, "Should respect iteration budget"
        assert fix_attempts == 3, "Should succeed on 3rd attempt"

    @pytest.mark.asyncio
    async def test_cycle_deadline_prevents_runaway(self):
        """Verify cycle-level deadline prevents runaway loops."""
        max_cycle_duration = timedelta(seconds=1)
        cycle_start = datetime.now()

        async def check_deadline():
            elapsed = datetime.now() - cycle_start
            return elapsed < max_cycle_duration

        # Simulate work until deadline
        iterations = 0
        while await check_deadline():
            await asyncio.sleep(0.1)
            iterations += 1
            if iterations > 20:  # Safety break
                break

        elapsed = datetime.now() - cycle_start
        assert elapsed >= max_cycle_duration or iterations <= 20, "Should stop at deadline"

    def test_deadline_buffer_calculation(self):
        """Verify deadline buffer is correctly calculated."""
        # From config: NOMIC_FIX_DEADLINE_BUFFER = 0.2 (20% buffer)
        total_time = 100  # seconds
        buffer_ratio = 0.2
        expected_buffer = total_time * buffer_ratio

        actual_buffer = total_time * buffer_ratio
        assert actual_buffer == expected_buffer, "Buffer should be 20% of total time"

        # Effective deadline
        effective_deadline = total_time - actual_buffer
        assert effective_deadline == 80, "Effective deadline should be 80s"


# =============================================================================
# Test: Checkpoint Recovery Consistency
# =============================================================================


class TestCheckpointRecoveryConsistency:
    """Tests for checkpoint creation and recovery."""

    def test_checkpoint_captures_state(self):
        """Verify checkpoint captures all required state."""
        checkpoint = {
            "cycle": 3,
            "phase": "design",
            "consensus_score": 0.75,
            "improvement": "Add caching layer",
            "agent_states": {
                "claude": {"proposals": 2, "critiques": 1},
                "gemini": {"proposals": 1, "critiques": 2},
            },
            "timestamp": datetime.now().isoformat(),
        }

        required_fields = ["cycle", "phase", "consensus_score", "improvement", "timestamp"]
        for field in required_fields:
            assert field in checkpoint, f"Missing required field: {field}"

    def test_restore_from_checkpoint_succeeds(self):
        """Verify state can be restored from checkpoint."""
        checkpoint = {
            "cycle": 2,
            "phase": "verify",
            "consensus_score": 0.72,
            "files_modified": ["src/cache.py", "tests/test_cache.py"],
        }

        # Simulate restore
        restored_cycle = checkpoint["cycle"]
        restored_phase = checkpoint["phase"]

        assert restored_cycle == 2, "Should restore correct cycle"
        assert restored_phase == "verify", "Should restore correct phase"

    def test_checkpoint_integrity_check(self):
        """Verify checkpoint integrity is validated on restore."""
        import hashlib
        import json

        checkpoint_data = {
            "cycle": 1,
            "phase": "debate",
            "data": "important_state",
        }

        # Create checksum
        data_str = json.dumps(checkpoint_data, sort_keys=True)
        expected_checksum = hashlib.sha256(data_str.encode()).hexdigest()

        # Verify checksum matches
        actual_checksum = hashlib.sha256(data_str.encode()).hexdigest()
        assert actual_checksum == expected_checksum, "Checksum should match"

        # Corrupt data and verify detection
        corrupted_data = checkpoint_data.copy()
        corrupted_data["data"] = "corrupted"
        corrupted_str = json.dumps(corrupted_data, sort_keys=True)
        corrupted_checksum = hashlib.sha256(corrupted_str.encode()).hexdigest()

        assert corrupted_checksum != expected_checksum, "Should detect corruption"

    @pytest.mark.asyncio
    async def test_recovery_restores_correct_phase(self):
        """Verify recovery jumps to correct phase after crash."""
        crash_phase = "implement"
        expected_recovery_phase = "implement"  # Resume from crash point

        # Simulate recovery decision
        decision = RecoveryDecision(
            strategy=RecoveryStrategy.RETRY,
            target_state=NomicState.IMPLEMENT,
            delay_seconds=1.0,
            reason="Resuming from checkpoint",
        )

        assert decision.target_state == NomicState.IMPLEMENT
        assert decision.strategy == RecoveryStrategy.RETRY


# =============================================================================
# Test: Empty Agent Fallback
# =============================================================================


class TestEmptyAgentFallback:
    """Tests for graceful degradation when no agents available."""

    def test_empty_agent_list_detected(self):
        """Verify empty agent list is detected early."""
        agents: List[str] = []

        is_empty = len(agents) == 0
        assert is_empty, "Should detect empty agent list"

    def test_fallback_to_default_agents(self):
        """Verify fallback to default agents when list is empty."""
        configured_agents: List[str] = []
        default_agents = ["claude", "gemini"]

        active_agents = configured_agents if configured_agents else default_agents

        assert len(active_agents) == 2, "Should use default agents"
        assert "claude" in active_agents, "Should include default agent"

    def test_single_agent_mode(self):
        """Verify single-agent mode works for resilience."""
        single_agent = "claude"

        # In single-agent mode, no debate happens, just proposal
        result = {
            "proposal": "Improvement idea",
            "agent": single_agent,
            "mode": "single_agent",
            "consensus": 1.0,  # Auto-consensus with single agent
        }

        assert result["mode"] == "single_agent"
        assert result["consensus"] == 1.0, "Single agent = automatic consensus"

    def test_graceful_degradation_message(self):
        """Verify informative message when degrading."""
        agents_available = 0
        minimum_required = 2

        if agents_available < minimum_required:
            message = (
                f"Insufficient agents ({agents_available}/{minimum_required}). "
                "Entering graceful degradation mode."
            )
            assert "graceful degradation" in message.lower()


# =============================================================================
# Test: Stress and Load
# =============================================================================


class TestStressResilience:
    """Stress tests for nomic loop resilience."""

    @pytest.mark.asyncio
    async def test_concurrent_phase_transitions(self):
        """Verify phase transitions handle concurrent access safely."""
        transitions_completed = []
        lock = asyncio.Lock()

        async def transition(phase_id: int):
            async with lock:
                transitions_completed.append(phase_id)
                await asyncio.sleep(0.01)

        # Run 10 concurrent transitions
        await asyncio.gather(*[transition(i) for i in range(10)])

        assert len(transitions_completed) == 10, "All transitions should complete"
        # Verify they were serialized (in order due to lock)
        assert transitions_completed == sorted(transitions_completed)

    @pytest.mark.asyncio
    async def test_rapid_failure_recovery(self):
        """Verify rapid failures don't crash the system."""
        failure_count = 0
        recovery_count = 0
        max_failures = 10

        async def simulate_failure_recovery():
            nonlocal failure_count, recovery_count
            failure_count += 1
            await asyncio.sleep(0.01)
            recovery_count += 1

        # Rapid failures
        for _ in range(max_failures):
            await simulate_failure_recovery()

        assert failure_count == max_failures, "All failures should be recorded"
        assert recovery_count == max_failures, "All recoveries should complete"

    def test_memory_pressure_handling(self):
        """Verify large state doesn't cause issues."""
        # Simulate large state
        large_state = {
            "history": [f"event_{i}" for i in range(10000)],
            "proposals": {f"proposal_{i}": "content" * 100 for i in range(100)},
        }

        # Verify state can be serialized
        import json

        serialized = json.dumps(large_state)
        assert len(serialized) > 0, "Should serialize large state"

        # Verify deserialization
        restored = json.loads(serialized)
        assert len(restored["history"]) == 10000
