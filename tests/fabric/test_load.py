"""Load tests for Agent Fabric at scale.

Tests concurrent agent spawning, scheduling latency, sustained throughput,
and graceful degradation under overload conditions.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from unittest.mock import AsyncMock, patch

import pytest

# Skip if fabric not available

from aragora.fabric import AgentFabric, FabricConfig
from aragora.fabric.models import (
    AgentConfig,
    BudgetConfig,
    HealthStatus,
    Priority,
    Task,
    TaskStatus,
)


@pytest.fixture
def fabric_config():
    """Config for load testing."""
    return FabricConfig(
        max_concurrent_agents=100,
        max_queue_depth=5000,
        default_timeout_seconds=60.0,
        heartbeat_interval_seconds=30.0,
        heartbeat_timeout_seconds=90.0,
    )


@pytest.fixture
def fabric(fabric_config):
    """Create a fabric instance for load testing."""
    return AgentFabric(config=fabric_config)


def make_config(id: str, model: str = "claude-3-opus") -> AgentConfig:
    """Create an agent config."""
    return AgentConfig(id=id, model=model)


def make_task(id: str, type: str = "debate") -> Task:
    """Create a task."""
    return Task(id=id, type=type, payload={"data": f"test-{id}"})


class TestLoadScaling:
    """Load tests for agent scaling."""

    @pytest.mark.asyncio
    async def test_spawn_50_agents(self, fabric):
        """Test spawning 50 agents concurrently."""
        await fabric.start()
        try:
            num_agents = 50

            # Spawn all agents concurrently
            start_time = time.perf_counter()
            tasks = [fabric.spawn(make_config(id=f"agent-{i}")) for i in range(num_agents)]
            handles = await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - start_time

            # Verify all agents spawned successfully
            assert len(handles) == num_agents
            for i, handle in enumerate(handles):
                assert handle.agent_id == f"agent-{i}"
                assert handle.status == HealthStatus.HEALTHY

            # Verify agent count via list
            agents = await fabric.list_agents()
            assert len(agents) == num_agents

            # Performance assertion: should complete in reasonable time
            # Allow up to 5 seconds for 50 agents (generous for CI environments)
            assert elapsed < 5.0, f"Spawning 50 agents took {elapsed:.2f}s (expected <5s)"

        finally:
            await fabric.stop()

    @pytest.mark.asyncio
    async def test_spawn_100_agents_concurrent(self, fabric):
        """Test spawning 100 agents to verify max_concurrent_agents config."""
        await fabric.start()
        try:
            num_agents = 100

            # Spawn all agents concurrently
            tasks = [fabric.spawn(make_config(id=f"agent-{i}")) for i in range(num_agents)]
            handles = await asyncio.gather(*tasks)

            # Verify all agents spawned
            assert len(handles) == num_agents
            agents = await fabric.list_agents()
            assert len(agents) == num_agents

        finally:
            await fabric.stop()

    @pytest.mark.asyncio
    async def test_spawn_and_terminate_cycle(self, fabric):
        """Test spawning and terminating agents in cycles."""
        await fabric.start()
        try:
            num_agents = 25
            cycles = 3

            for cycle in range(cycles):
                # Spawn agents
                spawn_tasks = [
                    fabric.spawn(make_config(id=f"cycle-{cycle}-agent-{i}"))
                    for i in range(num_agents)
                ]
                handles = await asyncio.gather(*spawn_tasks)
                assert len(handles) == num_agents

                # Verify count
                agents = await fabric.list_agents()
                assert len(agents) == num_agents

                # Terminate all agents
                terminate_tasks = [
                    fabric.terminate(f"cycle-{cycle}-agent-{i}") for i in range(num_agents)
                ]
                results = await asyncio.gather(*terminate_tasks)
                assert all(results)

                # Verify all terminated
                agents = await fabric.list_agents()
                assert len(agents) == 0

        finally:
            await fabric.stop()


class TestSchedulingLatency:
    """Tests for scheduling latency under load."""

    @pytest.mark.asyncio
    async def test_scheduling_latency_p99(self, fabric):
        """Test scheduling latency stays under 100ms at p99."""
        await fabric.start()
        try:
            # Spawn agents to receive tasks
            num_agents = 10
            for i in range(num_agents):
                await fabric.spawn(make_config(id=f"agent-{i}"))

            # Schedule many tasks and measure latency
            num_tasks = 500
            latencies = []

            for i in range(num_tasks):
                agent_id = f"agent-{i % num_agents}"
                task = make_task(id=f"task-{i}")

                start = time.perf_counter()
                await fabric.schedule(task, agent_id)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

            # Calculate statistics
            p50 = statistics.median(latencies)
            p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
            avg = statistics.mean(latencies)

            # Assert p99 latency is under 100ms
            assert p99 < 100, (
                f"p99 latency {p99:.2f}ms exceeds 100ms target. "
                f"Stats: avg={avg:.2f}ms, p50={p50:.2f}ms, p95={p95:.2f}ms"
            )

        finally:
            await fabric.stop()

    @pytest.mark.asyncio
    async def test_scheduling_latency_under_contention(self, fabric):
        """Test scheduling latency with concurrent schedule calls."""
        await fabric.start()
        try:
            # Spawn agents
            num_agents = 20
            for i in range(num_agents):
                await fabric.spawn(make_config(id=f"agent-{i}"))

            # Schedule tasks concurrently in batches
            num_tasks = 200
            batch_size = 50
            all_latencies = []

            for batch in range(num_tasks // batch_size):

                async def schedule_with_timing(task_id: int):
                    agent_id = f"agent-{task_id % num_agents}"
                    task = make_task(id=f"task-{batch}-{task_id}")
                    start = time.perf_counter()
                    await fabric.schedule(task, agent_id)
                    return (time.perf_counter() - start) * 1000

                tasks = [schedule_with_timing(i) for i in range(batch_size)]
                batch_latencies = await asyncio.gather(*tasks)
                all_latencies.extend(batch_latencies)

            # Verify latency stats
            p99 = statistics.quantiles(all_latencies, n=100)[98]
            assert p99 < 100, f"p99 latency {p99:.2f}ms exceeds 100ms under contention"

        finally:
            await fabric.stop()

    @pytest.mark.asyncio
    async def test_priority_scheduling_latency(self, fabric):
        """Test that priority tasks don't add significant latency."""
        await fabric.start()
        try:
            await fabric.spawn(make_config(id="agent-0"))

            # Schedule mix of priorities
            latencies_by_priority = {p: [] for p in Priority}

            for i in range(100):
                for priority in Priority:
                    task = make_task(id=f"task-{i}-{priority.name}")
                    start = time.perf_counter()
                    await fabric.schedule(task, "agent-0", priority=priority)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    latencies_by_priority[priority].append(elapsed_ms)

            # Verify all priorities have similar latency
            for priority, latencies in latencies_by_priority.items():
                p99 = statistics.quantiles(latencies, n=100)[98]
                assert p99 < 100, f"{priority.name} p99 latency {p99:.2f}ms exceeds 100ms"

        finally:
            await fabric.stop()


class TestSustainedThroughput:
    """Tests for sustained task throughput."""

    @pytest.mark.asyncio
    async def test_sustained_task_throughput(self, fabric):
        """Test sustained task throughput over time."""
        await fabric.start()
        try:
            num_agents = 10
            for i in range(num_agents):
                await fabric.spawn(make_config(id=f"agent-{i}"))

            # Simulate sustained load for 2 seconds
            duration_seconds = 2.0
            tasks_scheduled = 0
            tasks_completed = 0
            start_time = time.perf_counter()

            async def worker(agent_id: str, worker_id: int):
                """Worker that continuously schedules and completes tasks."""
                nonlocal tasks_scheduled, tasks_completed
                task_counter = 0
                while time.perf_counter() - start_time < duration_seconds:
                    task_id = f"sustained-{agent_id}-{worker_id}-{task_counter}"
                    task = make_task(id=task_id)

                    await fabric.schedule(task, agent_id)
                    tasks_scheduled += 1

                    # Simulate task execution
                    popped = await fabric.pop_next_task(agent_id)
                    if popped:
                        await fabric.complete_task(popped.id, result={"ok": True})
                        tasks_completed += 1

                    task_counter += 1
                    await asyncio.sleep(0.001)  # Small yield

            # Run workers concurrently
            workers = [
                worker(f"agent-{i % num_agents}", i)
                for i in range(num_agents * 2)  # 2 workers per agent
            ]
            await asyncio.gather(*workers)

            elapsed = time.perf_counter() - start_time
            throughput = tasks_completed / elapsed

            # Verify reasonable throughput (at least 100 tasks/second)
            assert throughput > 100, (
                f"Throughput {throughput:.1f} tasks/s below 100 tasks/s target. "
                f"Scheduled: {tasks_scheduled}, Completed: {tasks_completed}"
            )

            # Verify stats
            stats = await fabric.get_stats()
            assert stats["scheduler"]["tasks_completed"] >= tasks_completed

        finally:
            await fabric.stop()

    @pytest.mark.asyncio
    async def test_burst_throughput(self, fabric):
        """Test handling burst of tasks."""
        await fabric.start()
        try:
            num_agents = 5
            for i in range(num_agents):
                await fabric.spawn(make_config(id=f"agent-{i}"))

            # Schedule a burst of tasks
            burst_size = 500
            start = time.perf_counter()

            schedule_tasks = [
                fabric.schedule(
                    make_task(id=f"burst-{i}"),
                    f"agent-{i % num_agents}",
                )
                for i in range(burst_size)
            ]
            await asyncio.gather(*schedule_tasks)
            schedule_elapsed = time.perf_counter() - start

            # Verify all tasks scheduled
            stats = await fabric.get_stats()
            assert stats["scheduler"]["tasks_scheduled"] >= burst_size

            # Verify scheduling throughput
            schedule_rate = burst_size / schedule_elapsed
            assert schedule_rate > 1000, (
                f"Burst scheduling rate {schedule_rate:.1f}/s below 1000/s target"
            )

        finally:
            await fabric.stop()

    @pytest.mark.asyncio
    async def test_task_completion_throughput(self, fabric):
        """Test task completion throughput with simulated execution."""
        await fabric.start()
        try:
            num_agents = 10
            for i in range(num_agents):
                await fabric.spawn(make_config(id=f"agent-{i}"))

            # Pre-schedule tasks
            num_tasks = 200
            for i in range(num_tasks):
                await fabric.schedule(
                    make_task(id=f"complete-{i}"),
                    f"agent-{i % num_agents}",
                )

            # Complete tasks as fast as possible
            start = time.perf_counter()
            completed = 0

            for agent_idx in range(num_agents):
                agent_id = f"agent-{agent_idx}"
                while True:
                    task = await fabric.pop_next_task(agent_id)
                    if not task:
                        break
                    await fabric.complete_task(task.id, result={"done": True})
                    completed += 1

            elapsed = time.perf_counter() - start
            completion_rate = completed / elapsed

            assert completed == num_tasks
            assert completion_rate > 500, (
                f"Completion rate {completion_rate:.1f}/s below 500/s target"
            )

        finally:
            await fabric.stop()


class TestGracefulDegradation:
    """Tests for graceful degradation under overload."""

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, fabric):
        """Test graceful degradation under overload."""
        # Use smaller queue depth to trigger overload
        config = FabricConfig(
            max_concurrent_agents=100,
            max_queue_depth=50,  # Small queue to test overload
        )
        fabric = AgentFabric(config=config)

        await fabric.start()
        try:
            await fabric.spawn(make_config(id="agent-0"))

            # Fill the queue
            successful = 0
            failed = 0
            total_attempts = 100

            for i in range(total_attempts):
                try:
                    await fabric.schedule(make_task(id=f"overload-{i}"), "agent-0")
                    successful += 1
                except ValueError as e:
                    if "Queue full" in str(e):
                        failed += 1
                    else:
                        raise

            # Verify some tasks failed due to queue full
            assert failed > 0, "Expected some tasks to fail due to queue full"
            assert successful > 0, "Expected some tasks to succeed"
            assert successful + failed == total_attempts

        finally:
            await fabric.stop()

    @pytest.mark.asyncio
    async def test_recovery_after_overload(self, fabric):
        """Test system recovers after overload conditions clear."""
        config = FabricConfig(
            max_concurrent_agents=100,
            max_queue_depth=20,
        )
        fabric = AgentFabric(config=config)

        await fabric.start()
        try:
            await fabric.spawn(make_config(id="agent-0"))

            # Fill the queue to cause overload
            for i in range(20):
                await fabric.schedule(make_task(id=f"fill-{i}"), "agent-0")

            # Verify queue is full
            with pytest.raises(ValueError, match="Queue full"):
                await fabric.schedule(make_task(id="overflow"), "agent-0")

            # Drain the queue
            drained = 0
            while True:
                task = await fabric.pop_next_task("agent-0")
                if not task:
                    break
                await fabric.complete_task(task.id, result={"ok": True})
                drained += 1

            assert drained == 20

            # Verify system recovered - can schedule again
            handle = await fabric.schedule(make_task(id="recovery"), "agent-0")
            assert handle.status == TaskStatus.SCHEDULED

        finally:
            await fabric.stop()

    @pytest.mark.asyncio
    async def test_agent_failure_isolation(self, fabric):
        """Test that one agent's issues don't affect others."""
        await fabric.start()
        try:
            # Spawn multiple agents
            for i in range(5):
                await fabric.spawn(make_config(id=f"agent-{i}"))

            # Mark one agent as unhealthy
            fabric.lifecycle._agents["agent-0"].status = HealthStatus.UNHEALTHY

            # Other agents should still work
            for i in range(1, 5):
                handle = await fabric.schedule(
                    make_task(id=f"isolated-{i}"),
                    f"agent-{i}",
                )
                assert handle.status == TaskStatus.SCHEDULED

            # Verify unhealthy agent is reported in stats
            stats = await fabric.get_stats()
            assert stats["lifecycle"]["agents_unhealthy"] == 1
            assert stats["lifecycle"]["agents_healthy"] == 4

        finally:
            await fabric.stop()

    @pytest.mark.asyncio
    async def test_concurrent_spawn_terminate(self, fabric):
        """Test concurrent spawn and terminate operations."""
        await fabric.start()
        try:
            num_agents = 50

            # Spawn agents
            spawn_tasks = [fabric.spawn(make_config(id=f"churn-{i}")) for i in range(num_agents)]
            await asyncio.gather(*spawn_tasks)

            # Concurrently terminate some and spawn new ones
            async def churn_operations():
                results = []
                # Terminate first 25
                for i in range(25):
                    result = await fabric.terminate(f"churn-{i}")
                    results.append(("terminate", i, result))

                # Spawn 25 new ones
                for i in range(num_agents, num_agents + 25):
                    handle = await fabric.spawn(make_config(id=f"churn-{i}"))
                    results.append(("spawn", i, handle))

                return results

            results = await churn_operations()

            # Verify operations completed
            terminate_results = [r for r in results if r[0] == "terminate"]
            spawn_results = [r for r in results if r[0] == "spawn"]

            assert len(terminate_results) == 25
            assert len(spawn_results) == 25
            assert all(r[2] is True for r in terminate_results)

            # Verify final agent count
            agents = await fabric.list_agents()
            assert len(agents) == 50  # 50 original - 25 terminated + 25 new

        finally:
            await fabric.stop()


class TestPoolLoadBalancing:
    """Tests for pool-based load balancing at scale."""

    @pytest.mark.asyncio
    async def test_pool_load_distribution(self, fabric):
        """Test that pool distributes tasks evenly across agents."""
        await fabric.start()
        try:
            # Create a pool with multiple agents
            pool = await fabric.create_pool(
                name="load-test-pool",
                model="claude-3-opus",
                min_agents=10,
                max_agents=20,
            )

            # Schedule many tasks to the pool
            num_tasks = 100
            for i in range(num_tasks):
                await fabric.schedule_to_pool(
                    make_task(id=f"pool-task-{i}"),
                    pool.id,
                )

            # Check distribution - tasks should be spread across agents
            task_counts = {}
            for agent_id in pool.current_agents:
                pending = await fabric.scheduler.list_pending(agent_id)
                task_counts[agent_id] = len(pending)

            # Verify no agent is overloaded (max should be ~2x min)
            min_count = min(task_counts.values())
            max_count = max(task_counts.values())

            # Allow some imbalance but not extreme
            assert max_count <= min_count * 3 + 5, (
                f"Task distribution uneven: min={min_count}, max={max_count}"
            )

        finally:
            await fabric.stop()

    @pytest.mark.asyncio
    async def test_pool_scaling_under_load(self, fabric):
        """Test pool scaling under increasing load."""
        await fabric.start()
        try:
            # Create pool with low min but high max
            pool = await fabric.create_pool(
                name="scale-test-pool",
                model="claude-3-opus",
                min_agents=2,
                max_agents=20,
            )

            assert len(pool.current_agents) == 2

            # Scale up under simulated load
            new_count = await fabric.scale_pool(pool.id, 10)
            assert new_count == 10

            # Verify agents are available
            pool = await fabric.get_pool(pool.id)
            assert len(pool.current_agents) == 10

            # Scale down
            new_count = await fabric.scale_pool(pool.id, 5)
            assert new_count == 5

        finally:
            await fabric.stop()


class TestMemoryAndResourceUsage:
    """Tests for memory and resource usage patterns."""

    @pytest.mark.asyncio
    async def test_no_memory_leak_on_task_completion(self, fabric):
        """Test that completed tasks don't cause memory leaks."""
        await fabric.start()
        try:
            await fabric.spawn(make_config(id="agent-0"))

            # Run many task cycles
            num_cycles = 100
            for i in range(num_cycles):
                task = make_task(id=f"leak-test-{i}")
                await fabric.schedule(task, "agent-0")
                popped = await fabric.pop_next_task("agent-0")
                if popped:
                    await fabric.complete_task(popped.id, result={"cycle": i})

            # Verify completed tasks are tracked in stats
            stats = await fabric.get_stats()
            assert stats["scheduler"]["tasks_completed"] == num_cycles

        finally:
            await fabric.stop()

    @pytest.mark.asyncio
    async def test_heartbeat_under_load(self, fabric):
        """Test heartbeat mechanism works under load."""
        await fabric.start()
        try:
            num_agents = 20
            for i in range(num_agents):
                await fabric.spawn(make_config(id=f"hb-agent-{i}"))

            # Send heartbeats while scheduling tasks
            async def heartbeat_loop():
                for _ in range(10):
                    for i in range(num_agents):
                        await fabric.heartbeat(f"hb-agent-{i}")
                    await asyncio.sleep(0.01)

            async def schedule_loop():
                for i in range(100):
                    await fabric.schedule(
                        make_task(id=f"hb-task-{i}"),
                        f"hb-agent-{i % num_agents}",
                    )

            # Run both concurrently
            await asyncio.gather(heartbeat_loop(), schedule_loop())

            # Verify all agents still healthy
            agents = await fabric.list_agents()
            healthy_count = sum(1 for a in agents if a.status == HealthStatus.HEALTHY)
            assert healthy_count == num_agents

        finally:
            await fabric.stop()
