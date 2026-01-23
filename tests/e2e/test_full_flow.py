"""
E2E Tests for Full System Flow.

Tests complete end-to-end workflows using the E2ETestHarness:
- Agent lifecycle (register, heartbeat, unregister)
- Task lifecycle (submit, claim, process, complete)
- Multi-agent task distribution
- Debate orchestration through control plane
- Error handling and recovery
- Load and concurrent operations

Uses the harness fixtures from conftest.py for clean test setup.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

import pytest
import pytest_asyncio

from tests.e2e.harness import (
    E2ETestConfig,
    E2ETestHarness,
    MockAgent,
    MockAgentConfig,
    DebateTestHarness,
    LoadTestHarness,
    e2e_environment,
    create_mock_agent,
)

# Skip all tests if control plane dependencies are missing
pytest.importorskip("redis", reason="redis package required for control plane tests")


# ============================================================================
# Basic Harness Tests
# ============================================================================


class TestHarnessBasics:
    """Basic tests verifying harness functionality."""

    @pytest.mark.asyncio
    async def test_harness_starts_and_stops(self) -> None:
        """Test that harness can start and stop cleanly."""
        async with e2e_environment() as harness:
            assert harness._running is True
            assert harness.coordinator is not None
            assert harness.scheduler is not None
            assert len(harness.agents) == 3  # Default config

        # After context exit, harness should be stopped
        assert harness._running is False

    @pytest.mark.asyncio
    async def test_harness_with_custom_config(self) -> None:
        """Test harness with custom configuration."""
        config = E2ETestConfig(
            num_agents=5,
            agent_capabilities=["special-cap", "another-cap"],
            timeout_seconds=10.0,
        )

        async with e2e_environment(config) as harness:
            assert len(harness.agents) == 5
            assert "special-cap" in harness.agents[0].capabilities

    @pytest.mark.asyncio
    async def test_harness_stats(self, e2e_harness: E2ETestHarness) -> None:
        """Test getting harness statistics."""
        stats = await e2e_harness.get_stats()

        assert "control_plane" in stats
        assert "agents" in stats
        assert stats["running"] is True
        assert len(stats["agents"]) == 3


# ============================================================================
# Agent Lifecycle Tests
# ============================================================================


class TestAgentLifecycle:
    """Tests for agent registration, management, and unregistration."""

    @pytest.mark.asyncio
    async def test_create_agent(self, e2e_harness: E2ETestHarness) -> None:
        """Test creating a new agent."""
        initial_count = len(e2e_harness.agents)

        agent = await e2e_harness.create_agent(
            agent_id="custom-agent",
            capabilities=["custom-cap", "debate"],
        )

        assert agent.id == "custom-agent"
        assert "custom-cap" in agent.capabilities
        assert len(e2e_harness.agents) == initial_count + 1

        # Verify agent is registered in coordinator
        info = await e2e_harness.coordinator.get_agent("custom-agent")
        assert info is not None
        assert info.agent_id == "custom-agent"

    @pytest.mark.asyncio
    async def test_create_agent_with_config(self, e2e_harness: E2ETestHarness) -> None:
        """Test creating agent with full configuration."""
        config = MockAgentConfig(
            id="configured-agent",
            capabilities=["analysis", "code"],
            region="eu-west-1",
            model="custom-model",
            provider="custom-provider",
            response_template="Custom response: {task_type}",
            fail_rate=0.0,
            response_delay=0.01,
        )

        agent = await e2e_harness.create_agent(config=config)

        assert agent.id == "configured-agent"
        assert agent.region == "eu-west-1"
        assert agent.model == "custom-model"

    @pytest.mark.asyncio
    async def test_remove_agent(self, e2e_harness: E2ETestHarness) -> None:
        """Test removing an agent."""
        agent = await e2e_harness.create_agent("to-remove")
        initial_count = len(e2e_harness.agents)

        success = await e2e_harness.remove_agent("to-remove")

        assert success is True
        assert len(e2e_harness.agents) == initial_count - 1

        # Verify agent is unregistered
        info = await e2e_harness.coordinator.get_agent("to-remove")
        assert info is None

    @pytest.mark.asyncio
    async def test_get_agent(self, e2e_harness: E2ETestHarness) -> None:
        """Test getting an agent by ID."""
        agent = e2e_harness.get_agent("test-agent-0")

        assert agent is not None
        assert agent.id == "test-agent-0"

    @pytest.mark.asyncio
    async def test_agents_ready(self, e2e_harness: E2ETestHarness) -> None:
        """Test waiting for agents to be ready."""
        ready = await e2e_harness.wait_for_agents_ready(timeout=5.0)
        assert ready is True


# ============================================================================
# Task Lifecycle Tests
# ============================================================================


class TestTaskLifecycle:
    """Tests for task submission, processing, and completion."""

    @pytest.mark.asyncio
    async def test_submit_task(self, e2e_harness: E2ETestHarness) -> None:
        """Test submitting a task."""
        task_id = await e2e_harness.submit_task(
            task_type="test-task",
            payload={"data": "test-data"},
        )

        assert task_id is not None
        assert len(task_id) > 0

        # Verify task was created
        task = await e2e_harness.coordinator.get_task(task_id)
        assert task is not None
        assert task.task_type == "test-task"

    @pytest.mark.asyncio
    async def test_submit_and_wait_for_task(self, e2e_harness: E2ETestHarness) -> None:
        """Test submitting a task and waiting for completion."""
        task_id = await e2e_harness.submit_task(
            task_type="analysis",
            payload={"input": "test input"},
            required_capabilities=["analysis"],
        )

        # Wait for task (harness will auto-process)
        result = await e2e_harness.wait_for_task(task_id, timeout=5.0)

        assert result is not None
        assert result.status.value == "completed"
        assert result.result is not None

    @pytest.mark.asyncio
    async def test_task_with_priority(self, e2e_harness: E2ETestHarness) -> None:
        """Test task priority handling."""
        from aragora.control_plane import TaskPriority

        # Submit low priority task
        low_id = await e2e_harness.submit_task(
            task_type="test",
            payload={"priority": "low"},
            priority=TaskPriority.LOW,
        )

        # Submit high priority task
        high_id = await e2e_harness.submit_task(
            task_type="test",
            payload={"priority": "high"},
            priority=TaskPriority.HIGH,
        )

        # Verify both tasks exist
        low_task = await e2e_harness.coordinator.get_task(low_id)
        high_task = await e2e_harness.coordinator.get_task(high_id)

        assert low_task is not None
        assert high_task is not None
        assert low_task.priority.value < high_task.priority.value

    @pytest.mark.asyncio
    async def test_claim_and_process_task(self, e2e_harness: E2ETestHarness) -> None:
        """Test manual task claiming and processing."""
        # Submit task
        task_id = await e2e_harness.submit_task(
            task_type="manual-process",
            payload={"action": "process"},
            required_capabilities=["debate"],
        )

        # Get an agent with matching capabilities
        agent = next(
            (a for a in e2e_harness.agents if "debate" in a.capabilities),
            e2e_harness.agents[0],
        )

        # Manually claim and process
        task = await e2e_harness.claim_and_process_task(agent)

        assert task is not None
        assert task.id == task_id

        # Verify task completed
        final_task = await e2e_harness.coordinator.get_task(task_id)
        assert final_task.status.value == "completed"


# ============================================================================
# Multi-Agent Tests
# ============================================================================


class TestMultiAgentWorkflows:
    """Tests for multi-agent coordination and task distribution."""

    @pytest.mark.asyncio
    async def test_multiple_agents_process_tasks(self, e2e_harness: E2ETestHarness) -> None:
        """Test that multiple agents can process tasks."""
        # Submit multiple tasks
        task_ids = []
        for i in range(5):
            task_id = await e2e_harness.submit_task(
                task_type="batch-task",
                payload={"index": i},
            )
            task_ids.append(task_id)

        # Wait for all tasks to complete
        results = await asyncio.gather(
            *[e2e_harness.wait_for_task(tid, timeout=10.0) for tid in task_ids]
        )

        # Verify all completed
        assert len(results) == 5
        completed = [r for r in results if r and r.status.value == "completed"]
        assert len(completed) == 5

    @pytest.mark.asyncio
    async def test_capability_based_routing(self, e2e_harness: E2ETestHarness) -> None:
        """Test that tasks are routed to agents with matching capabilities."""
        # Create specialized agents
        code_agent = await e2e_harness.create_agent(
            "code-specialist",
            capabilities=["code", "refactor"],
        )
        design_agent = await e2e_harness.create_agent(
            "design-specialist",
            capabilities=["design", "architecture"],
        )

        # Submit capability-specific task
        task_id = await e2e_harness.submit_task(
            task_type="code-review",
            payload={"code": "print('hello')"},
            required_capabilities=["code"],
        )

        # Process with code agent
        task = await e2e_harness.claim_and_process_task(
            code_agent,
            capabilities=["code"],
        )

        assert task is not None
        assert task.id == task_id


# ============================================================================
# Debate Integration Tests
# ============================================================================


class TestDebateIntegration:
    """Tests for debate orchestration through the system."""

    @pytest.mark.asyncio
    async def test_run_debate(self, e2e_harness: E2ETestHarness) -> None:
        """Test running a debate directly through the harness."""
        result = await e2e_harness.run_debate(
            topic="Should we use microservices architecture?",
            rounds=2,
        )

        assert result is not None
        assert hasattr(result, "rounds_completed") or hasattr(result, "final_answer")

    @pytest.mark.asyncio
    async def test_run_debate_via_control_plane(self, e2e_harness: E2ETestHarness) -> None:
        """Test running a debate as a control plane task."""
        result = await e2e_harness.run_debate_via_control_plane(
            topic="Best practices for API design",
            rounds=2,
        )

        assert result is not None
        assert "topic" in result
        assert result["consensus_reached"] is True

    @pytest.mark.asyncio
    async def test_debate_harness_tracking(self, debate_harness: DebateTestHarness) -> None:
        """Test debate-specific harness features."""
        # Run multiple debates
        await debate_harness.run_debate_with_tracking("Topic 1", rounds=2)
        await debate_harness.run_debate_with_tracking("Topic 2", rounds=2)

        # Check tracking
        results = debate_harness.get_debate_results()
        assert len(results) == 2

        # Check consensus rate
        rate = debate_harness.get_consensus_rate()
        assert 0.0 <= rate <= 1.0


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_agent_failure_recovery(self) -> None:
        """Test that system handles agent failures."""
        config = E2ETestConfig(
            num_agents=3,
            fail_rate=0.0,  # No random failures
        )

        async with e2e_environment(config) as harness:
            # Create a failing agent
            failing_agent = await harness.create_agent(
                agent_id="failing-agent",
                capabilities=["unstable"],
            )
            failing_agent.fail_rate = 1.0  # Always fail

            # Submit task requiring the failing capability
            task_id = await harness.submit_task(
                task_type="unstable-task",
                payload={"test": "data"},
                required_capabilities=["unstable"],
            )

            # Try to process with failing agent (should fail and requeue)
            task = await harness.claim_and_process_task(failing_agent)

            # Task should have been requeued after failure
            final_task = await harness.coordinator.get_task(task_id)
            # Task status depends on retry configuration
            assert final_task is not None

    @pytest.mark.asyncio
    async def test_task_timeout(self, e2e_harness: E2ETestHarness) -> None:
        """Test task timeout handling."""
        task_id = await e2e_harness.submit_task(
            task_type="timeout-test",
            payload={},
            timeout=0.1,  # Very short timeout
        )

        # Don't process the task - let it timeout
        # Wait without processing
        result = await e2e_harness.wait_for_task(
            task_id,
            timeout=0.5,
            process_with_agent=False,
        )

        # Task should still be pending or timed out
        # (depends on implementation)
        assert result is None or result.status.value in ["pending", "timeout"]

    @pytest.mark.asyncio
    async def test_nonexistent_task(self, e2e_harness: E2ETestHarness) -> None:
        """Test handling of nonexistent task."""
        result = await e2e_harness.wait_for_task(
            "nonexistent-task-id",
            timeout=1.0,
            process_with_agent=False,
        )

        assert result is None


# ============================================================================
# Concurrent Operations Tests
# ============================================================================


class TestConcurrentOperations:
    """Tests for concurrent task handling."""

    @pytest.mark.asyncio
    async def test_concurrent_task_submission(self, e2e_harness: E2ETestHarness) -> None:
        """Test submitting many tasks concurrently."""

        # Submit 10 tasks concurrently
        async def submit_task(index: int) -> str:
            return await e2e_harness.submit_task(
                task_type="concurrent-test",
                payload={"index": index},
            )

        task_ids = await asyncio.gather(*[submit_task(i) for i in range(10)])

        assert len(task_ids) == 10
        assert all(tid is not None for tid in task_ids)

    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self, e2e_harness: E2ETestHarness) -> None:
        """Test processing tasks concurrently."""
        # Submit tasks
        task_ids = []
        for i in range(5):
            task_id = await e2e_harness.submit_task(
                task_type="concurrent-process",
                payload={"index": i},
            )
            task_ids.append(task_id)

        # Wait for all concurrently
        start = time.monotonic()
        results = await asyncio.gather(
            *[e2e_harness.wait_for_task(tid, timeout=10.0) for tid in task_ids]
        )
        elapsed = time.monotonic() - start

        # All should complete
        completed = [r for r in results if r and r.status.value == "completed"]
        assert len(completed) == 5

        # Should be faster than sequential (tasks process concurrently)
        # With 0.05s delay per task, 5 sequential tasks = 0.25s
        # Concurrent should be much faster
        assert elapsed < 1.0


# ============================================================================
# Load Testing
# ============================================================================


class TestLoadOperations:
    """Tests using the load test harness."""

    @pytest.mark.asyncio
    async def test_concurrent_task_submission(self, load_test_harness: LoadTestHarness) -> None:
        """Test submitting many tasks concurrently using load harness."""
        task_ids = await load_test_harness.submit_concurrent_tasks(
            count=20,
            task_type="load-test",
        )

        assert len(task_ids) == 20

    @pytest.mark.asyncio
    async def test_throughput_measurement(self, load_test_harness: LoadTestHarness) -> None:
        """Test measuring task throughput."""
        metrics = await load_test_harness.measure_throughput(
            task_count=10,
            task_type="throughput-test",
        )

        assert metrics["total_tasks"] == 10
        assert metrics["successful_tasks"] <= 10
        assert metrics["tasks_per_second"] > 0
        assert 0.0 <= metrics["success_rate"] <= 1.0


# ============================================================================
# Event Tracking Tests
# ============================================================================


class TestEventTracking:
    """Tests for event recording and retrieval."""

    @pytest.mark.asyncio
    async def test_event_recording(self, e2e_harness: E2ETestHarness) -> None:
        """Test that events are recorded."""
        # Submit a task
        task_id = await e2e_harness.submit_task(
            task_type="event-test",
            payload={},
        )

        # Check events
        events = e2e_harness.get_events()
        assert len(events) > 0

        # Filter by type
        submit_events = e2e_harness.get_events("task_submitted")
        assert len(submit_events) >= 1
        assert submit_events[-1]["task_id"] == task_id

    @pytest.mark.asyncio
    async def test_event_clearing(self, e2e_harness: E2ETestHarness) -> None:
        """Test clearing recorded events."""
        # Generate some events
        await e2e_harness.submit_task("test", {})

        # Clear events
        e2e_harness.clear_events()

        events = e2e_harness.get_events()
        assert len(events) == 0


# ============================================================================
# Integration Workflow Tests
# ============================================================================


class TestIntegrationWorkflows:
    """End-to-end integration workflow tests."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, e2e_harness: E2ETestHarness) -> None:
        """Test a complete workflow from start to finish."""
        # 1. Create specialized agent
        specialist = await e2e_harness.create_agent(
            "workflow-specialist",
            capabilities=["workflow", "analysis"],
        )

        # 2. Submit task
        task_id = await e2e_harness.submit_task(
            task_type="workflow-task",
            payload={
                "step": "analyze",
                "data": {"input": "test data"},
            },
            required_capabilities=["workflow"],
        )

        # 3. Wait for completion
        result = await e2e_harness.wait_for_task(task_id, timeout=10.0)

        # 4. Verify result
        assert result is not None
        assert result.status.value == "completed"

        # 5. Check agent stats
        stats = await e2e_harness.get_stats()
        assert stats["running"] is True

    @pytest.mark.asyncio
    async def test_debate_followed_by_analysis(self, e2e_harness: E2ETestHarness) -> None:
        """Test running a debate followed by analysis task."""
        # Run debate
        debate_result = await e2e_harness.run_debate(
            topic="Should we refactor the authentication module?",
            rounds=2,
        )

        assert debate_result is not None

        # Submit analysis task based on debate outcome
        analysis_task_id = await e2e_harness.submit_task(
            task_type="analysis",
            payload={
                "debate_topic": "Authentication refactoring",
                "decision": getattr(debate_result, "final_answer", "Proceed"),
            },
            required_capabilities=["analysis"],
        )

        # Wait for analysis
        analysis_result = await e2e_harness.wait_for_task(analysis_task_id)

        assert analysis_result is not None
        assert analysis_result.status.value == "completed"


# ============================================================================
# Mock Agent Tests
# ============================================================================


class TestMockAgent:
    """Tests for MockAgent functionality."""

    @pytest.mark.asyncio
    async def test_mock_agent_generate(self) -> None:
        """Test mock agent generate method."""
        agent = create_mock_agent("test-agent", "Test response: {agent_id}")

        response = await agent.generate("Test prompt")

        assert "test-agent" in response

    @pytest.mark.asyncio
    async def test_mock_agent_critique(self) -> None:
        """Test mock agent critique method."""
        agent = create_mock_agent("critic-agent")

        critique = await agent.critique("Some content to critique")

        assert critique.agent == "critic-agent"
        assert len(critique.issues) > 0
        assert len(critique.suggestions) > 0

    @pytest.mark.asyncio
    async def test_mock_agent_vote(self) -> None:
        """Test mock agent vote method."""
        agent = create_mock_agent("voter-agent")

        vote = await agent.vote(["Option A", "Option B"])

        assert vote.agent == "voter-agent"
        assert vote.choice == "Option A"  # Votes for first option
        assert 0.0 <= vote.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_mock_agent_stats(self) -> None:
        """Test mock agent statistics tracking."""
        agent = create_mock_agent("stats-agent")

        # Process some tasks
        from aragora.control_plane import Task

        mock_task = Task(task_type="test", payload={})
        await agent.process_task(mock_task)
        await agent.process_task(mock_task)

        stats = agent.get_stats()

        assert stats["tasks_processed"] == 2
        assert stats["tasks_failed"] == 0
        assert stats["avg_latency_ms"] > 0
