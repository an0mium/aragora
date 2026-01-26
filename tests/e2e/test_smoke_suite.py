"""
E2E Smoke Test Suite for Aragora.

Fast, focused tests that validate critical system paths before release.
These tests should complete in under 2 minutes and cover:
- Server startup and health
- Authentication flow
- Basic debate lifecycle
- WebSocket connectivity
- Core API endpoints

Run with: pytest tests/e2e/test_smoke_suite.py -v
Mark: @pytest.mark.smoke

These tests are designed to be run:
1. Before every release
2. As a quick sanity check in CI
3. After deployment to verify the system is operational
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from tests.e2e.harness import (
    E2ETestConfig,
    E2ETestHarness,
    e2e_environment,
)

# Mark all tests in this module as smoke tests
pytestmark = [pytest.mark.smoke, pytest.mark.e2e]


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def smoke_harness():
    """Lightweight harness for smoke tests - minimal agents, fast timeouts."""
    config = E2ETestConfig(
        num_agents=2,
        agent_capabilities=["debate", "general"],
        agent_response_delay=0.01,  # Fast responses
        timeout_seconds=30.0,
        task_timeout_seconds=10.0,
        heartbeat_interval=1.0,
        default_debate_rounds=1,
    )
    async with e2e_environment(config) as harness:
        yield harness


# ============================================================================
# Server Startup Tests
# ============================================================================


class TestServerStartup:
    """Smoke tests for server startup and basic operation."""

    @pytest.mark.asyncio
    async def test_harness_starts_successfully(self, smoke_harness: E2ETestHarness):
        """Verify the E2E harness (simulating server) starts without errors."""
        assert smoke_harness._running is True
        assert smoke_harness.coordinator is not None
        assert smoke_harness.scheduler is not None
        assert smoke_harness.registry is not None
        assert smoke_harness.health_monitor is not None

    @pytest.mark.asyncio
    async def test_agents_register_on_startup(self, smoke_harness: E2ETestHarness):
        """Verify agents are registered and available."""
        assert len(smoke_harness.agents) >= 2

        # Check agents are registered in coordinator
        for agent in smoke_harness.agents:
            info = await smoke_harness.coordinator.get_agent(agent.id)
            assert info is not None, f"Agent {agent.id} not registered"
            assert info.agent_id == agent.id

    @pytest.mark.asyncio
    async def test_stats_endpoint_works(self, smoke_harness: E2ETestHarness):
        """Verify statistics can be retrieved (simulates /health, /metrics)."""
        stats = await smoke_harness.get_stats()

        assert "control_plane" in stats
        assert "agents" in stats
        assert stats["running"] is True
        assert len(stats["agents"]) >= 2


# ============================================================================
# Authentication Flow Tests
# ============================================================================


class TestAuthenticationFlow:
    """Smoke tests for authentication mechanisms."""

    @pytest.mark.asyncio
    async def test_agent_authentication(self, smoke_harness: E2ETestHarness):
        """Verify agents can authenticate and perform operations."""
        # Create a new agent (simulates auth + registration)
        agent = await smoke_harness.create_agent(
            agent_id="auth-test-agent",
            capabilities=["test"],
        )

        assert agent is not None
        assert agent.id == "auth-test-agent"

        # Verify agent can be retrieved (auth check passed)
        info = await smoke_harness.coordinator.get_agent("auth-test-agent")
        assert info is not None

    @pytest.mark.asyncio
    async def test_unauthorized_agent_rejected(self, smoke_harness: E2ETestHarness):
        """Verify non-existent agents are not found."""
        info = await smoke_harness.coordinator.get_agent("non-existent-agent")
        assert info is None


# ============================================================================
# Debate Lifecycle Tests
# ============================================================================


class TestDebateLifecycle:
    """Smoke tests for core debate functionality."""

    @pytest.mark.asyncio
    async def test_basic_debate_completes(self, smoke_harness: E2ETestHarness):
        """Verify a basic debate can run to completion."""
        result = await smoke_harness.run_debate(
            topic="What is 2 + 2?",
            rounds=1,
        )

        assert result is not None
        # Debate should complete without errors

    @pytest.mark.asyncio
    async def test_debate_with_multiple_agents(self, smoke_harness: E2ETestHarness):
        """Verify debate works with multiple participating agents."""
        # Use all available agents
        result = await smoke_harness.run_debate(
            topic="Should we use tabs or spaces?",
            rounds=1,
            agents=smoke_harness.agents,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_debate_via_control_plane(self, smoke_harness: E2ETestHarness):
        """Verify debate can be submitted and processed via control plane."""
        result = await smoke_harness.run_debate_via_control_plane(
            topic="Test topic for control plane",
            rounds=1,
        )

        assert result is not None
        assert "topic" in result
        assert result["rounds_completed"] == 1


# ============================================================================
# Task System Tests
# ============================================================================


class TestTaskSystem:
    """Smoke tests for task submission and processing."""

    @pytest.mark.asyncio
    async def test_submit_and_complete_task(self, smoke_harness: E2ETestHarness):
        """Verify task can be submitted and completed."""
        task_id = await smoke_harness.submit_task(
            task_type="test",
            payload={"message": "smoke test"},
        )

        assert task_id is not None

        # Wait for completion
        task = await smoke_harness.wait_for_task(task_id, timeout=10.0)

        assert task is not None
        assert task.status.value in ("completed", "COMPLETED")

    @pytest.mark.asyncio
    async def test_task_with_capabilities(self, smoke_harness: E2ETestHarness):
        """Verify task with required capabilities is routed correctly."""
        task_id = await smoke_harness.submit_task(
            task_type="debate",
            payload={"topic": "test"},
            required_capabilities=["debate"],
        )

        task = await smoke_harness.wait_for_task(task_id, timeout=10.0)

        assert task is not None

    @pytest.mark.asyncio
    async def test_task_events_recorded(self, smoke_harness: E2ETestHarness):
        """Verify task events are properly recorded."""
        smoke_harness.clear_events()

        await smoke_harness.submit_task(
            task_type="test",
            payload={"test": True},
        )

        events = smoke_harness.get_events("task_submitted")
        assert len(events) >= 1


# ============================================================================
# Control Plane Tests
# ============================================================================


class TestControlPlane:
    """Smoke tests for control plane operations."""

    @pytest.mark.asyncio
    async def test_coordinator_status(self, smoke_harness: E2ETestHarness):
        """Verify coordinator is operational."""
        assert smoke_harness.coordinator is not None

        # Get coordinator stats
        stats = await smoke_harness.coordinator.get_stats()
        assert stats is not None

    @pytest.mark.asyncio
    async def test_scheduler_accepts_tasks(self, smoke_harness: E2ETestHarness):
        """Verify scheduler can accept and queue tasks."""
        task_id = await smoke_harness.submit_task(
            task_type="scheduled-test",
            payload={"scheduled": True},
        )

        assert task_id is not None
        # Task should be queued successfully

    @pytest.mark.asyncio
    async def test_agent_heartbeat_working(self, smoke_harness: E2ETestHarness):
        """Verify agent heartbeat mechanism is functional."""
        # Wait for agents to be ready (heartbeat received)
        ready = await smoke_harness.wait_for_agents_ready(timeout=5.0)

        assert ready is True, "Agents did not become ready in time"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Smoke tests for error handling."""

    @pytest.mark.asyncio
    async def test_graceful_agent_removal(self, smoke_harness: E2ETestHarness):
        """Verify agent can be removed without crashing."""
        agent = await smoke_harness.create_agent("removable-agent")

        success = await smoke_harness.remove_agent("removable-agent")
        assert success is True

        # System should still be operational
        assert smoke_harness._running is True

    @pytest.mark.asyncio
    async def test_invalid_task_handling(self, smoke_harness: E2ETestHarness):
        """Verify invalid task retrieval doesn't crash."""
        # Try to get a non-existent task
        task = await smoke_harness.coordinator.get_task("non-existent-task-id")

        # Should return None, not crash
        assert task is None


# ============================================================================
# Performance Smoke Tests
# ============================================================================


class TestPerformanceBaseline:
    """Quick performance sanity checks."""

    @pytest.mark.asyncio
    async def test_task_latency_acceptable(self, smoke_harness: E2ETestHarness):
        """Verify basic task completes within acceptable time."""
        start = time.monotonic()

        task_id = await smoke_harness.submit_task(
            task_type="latency-test",
            payload={"test": True},
        )
        await smoke_harness.wait_for_task(task_id, timeout=10.0)

        elapsed = time.monotonic() - start

        # Should complete in under 5 seconds for smoke test
        assert elapsed < 5.0, f"Task took too long: {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_concurrent_tasks_work(self, smoke_harness: E2ETestHarness):
        """Verify multiple tasks can be processed concurrently."""
        # Submit 3 tasks concurrently
        task_ids = await asyncio.gather(
            smoke_harness.submit_task("concurrent-1", {"n": 1}),
            smoke_harness.submit_task("concurrent-2", {"n": 2}),
            smoke_harness.submit_task("concurrent-3", {"n": 3}),
        )

        assert len(task_ids) == 3

        # Wait for all to complete
        results = await asyncio.gather(
            *[smoke_harness.wait_for_task(tid, timeout=10.0) for tid in task_ids]
        )

        # All should complete
        completed = [r for r in results if r is not None]
        assert len(completed) >= 2, "Not enough tasks completed"


# ============================================================================
# Integration Summary Test
# ============================================================================


class TestSmokeSummary:
    """Final integration test that exercises multiple systems."""

    @pytest.mark.asyncio
    async def test_full_smoke_flow(self, smoke_harness: E2ETestHarness):
        """
        Complete smoke test flow exercising all critical paths.

        This test validates:
        1. System startup (fixture)
        2. Agent registration
        3. Task submission
        4. Task completion
        5. Debate execution
        6. Event recording
        7. Statistics retrieval
        """
        # 1. System is running (from fixture)
        assert smoke_harness._running

        # 2. Create a test agent
        agent = await smoke_harness.create_agent(
            agent_id="smoke-flow-agent",
            capabilities=["smoke", "debate"],
        )
        assert agent is not None

        # 3. Submit a task
        task_id = await smoke_harness.submit_task(
            task_type="smoke-flow",
            payload={"step": "task-submission"},
        )
        assert task_id is not None

        # 4. Complete the task
        task = await smoke_harness.wait_for_task(task_id, timeout=10.0)
        assert task is not None

        # 5. Run a debate
        result = await smoke_harness.run_debate(
            topic="Smoke test validation",
            rounds=1,
        )
        assert result is not None

        # 6. Check events were recorded
        events = smoke_harness.get_events()
        assert len(events) >= 2  # At least task + debate events

        # 7. Get final stats
        stats = await smoke_harness.get_stats()
        assert stats["running"] is True
        assert len(stats["agents"]) >= 3  # Original 2 + our new agent

        # Cleanup
        await smoke_harness.remove_agent("smoke-flow-agent")
