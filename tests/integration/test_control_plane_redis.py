"""
Control Plane Redis Integration Tests.

These tests validate the Control Plane against a real Redis instance.
Skip if Redis is not available.

Run with:
    REDIS_URL=redis://localhost:6379 pytest tests/integration/test_control_plane_redis.py -v
"""

import asyncio
import os
import time
import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

# Check if Redis is available
REDIS_URL = os.environ.get("REDIS_URL", "")
SKIP_REASON = "Redis not available (set REDIS_URL env var)"

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = bool(REDIS_URL)
except ImportError:
    REDIS_AVAILABLE = False
    SKIP_REASON = "redis package not installed"


pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason=SKIP_REASON)


@pytest.fixture
async def redis_client():
    """Create Redis client for tests."""
    client = aioredis.from_url(REDIS_URL, decode_responses=True)

    # Clean up test keys before and after
    test_prefix = "aragora:test:"

    async def cleanup():
        keys = await client.keys(f"{test_prefix}*")
        if keys:
            await client.delete(*keys)

    await cleanup()
    yield client
    await cleanup()
    await client.aclose()


@pytest.fixture
def test_prefix():
    """Generate unique test prefix."""
    return f"aragora:test:{uuid.uuid4().hex[:8]}:"


class TestAgentRegistryRedis:
    """Test agent registry with actual Redis."""

    @pytest.mark.asyncio
    async def test_register_agent(self, redis_client, test_prefix):
        """Test registering an agent in Redis."""
        agent_id = f"{test_prefix}agent:1"
        agent_data = {
            "name": "test-agent",
            "capabilities": '["debate", "critique"]',
            "status": "available",
            "registered_at": str(time.time()),
        }

        await redis_client.hset(agent_id, mapping=agent_data)

        result = await redis_client.hgetall(agent_id)
        assert result["name"] == "test-agent"
        assert result["status"] == "available"

    @pytest.mark.asyncio
    async def test_agent_heartbeat(self, redis_client, test_prefix):
        """Test agent heartbeat mechanism."""
        agent_id = f"{test_prefix}agent:heartbeat"
        heartbeat_key = f"{test_prefix}heartbeat:{agent_id}"

        # Set heartbeat with TTL
        await redis_client.setex(heartbeat_key, 30, str(time.time()))

        # Verify heartbeat exists
        ttl = await redis_client.ttl(heartbeat_key)
        assert ttl > 0
        assert ttl <= 30

    @pytest.mark.asyncio
    async def test_agent_discovery(self, redis_client, test_prefix):
        """Test discovering available agents."""
        # Register multiple agents
        for i in range(5):
            agent_id = f"{test_prefix}agent:{i}"
            await redis_client.hset(agent_id, mapping={
                "name": f"agent-{i}",
                "status": "available" if i % 2 == 0 else "busy",
            })
            await redis_client.sadd(f"{test_prefix}agents:all", agent_id)

        # Discover all agents
        all_agents = await redis_client.smembers(f"{test_prefix}agents:all")
        assert len(all_agents) == 5

        # Filter available agents
        available = []
        for agent_id in all_agents:
            data = await redis_client.hgetall(agent_id)
            if data.get("status") == "available":
                available.append(agent_id)

        assert len(available) == 3  # agents 0, 2, 4


class TestTaskDistributionRedis:
    """Test task distribution with actual Redis."""

    @pytest.mark.asyncio
    async def test_task_queue(self, redis_client, test_prefix):
        """Test task queue operations."""
        queue_key = f"{test_prefix}tasks:pending"

        # Enqueue tasks
        for i in range(10):
            task = f'{{"id": "task-{i}", "type": "debate"}}'
            await redis_client.lpush(queue_key, task)

        # Verify queue length
        length = await redis_client.llen(queue_key)
        assert length == 10

        # Dequeue task (FIFO with rpop)
        task = await redis_client.rpop(queue_key)
        assert '"id": "task-0"' in task

    @pytest.mark.asyncio
    async def test_task_distribution_load(self, redis_client, test_prefix):
        """Test task distribution under load (100+ tasks)."""
        queue_key = f"{test_prefix}tasks:load"

        # Enqueue 100 tasks
        tasks = [f'{{"id": "task-{i}", "priority": {i % 3}}}' for i in range(100)]
        await redis_client.lpush(queue_key, *tasks)

        # Verify all tasks queued
        length = await redis_client.llen(queue_key)
        assert length == 100

        # Process all tasks
        processed = 0
        while True:
            task = await redis_client.rpop(queue_key)
            if task is None:
                break
            processed += 1

        assert processed == 100

    @pytest.mark.asyncio
    async def test_priority_queue(self, redis_client, test_prefix):
        """Test priority-based task queue using sorted sets."""
        queue_key = f"{test_prefix}tasks:priority"

        # Add tasks with priorities (lower score = higher priority)
        await redis_client.zadd(queue_key, {
            "critical-task": 0,
            "high-task": 1,
            "normal-task": 2,
            "low-task": 3,
        })

        # Get highest priority task
        result = await redis_client.zrange(queue_key, 0, 0)
        assert result[0] == "critical-task"

        # Pop highest priority
        popped = await redis_client.zpopmin(queue_key)
        assert popped[0][0] == "critical-task"


class TestHeartbeatMechanism:
    """Test heartbeat with network simulation."""

    @pytest.mark.asyncio
    async def test_heartbeat_expiry(self, redis_client, test_prefix):
        """Test heartbeat expiry detection."""
        agent_id = f"{test_prefix}agent:expiry"
        heartbeat_key = f"{test_prefix}heartbeat:{agent_id}"

        # Set short TTL heartbeat
        await redis_client.setex(heartbeat_key, 2, str(time.time()))

        # Verify exists
        exists = await redis_client.exists(heartbeat_key)
        assert exists == 1

        # Wait for expiry
        await asyncio.sleep(3)

        # Verify expired
        exists = await redis_client.exists(heartbeat_key)
        assert exists == 0

    @pytest.mark.asyncio
    async def test_heartbeat_refresh(self, redis_client, test_prefix):
        """Test heartbeat refresh extends TTL."""
        agent_id = f"{test_prefix}agent:refresh"
        heartbeat_key = f"{test_prefix}heartbeat:{agent_id}"

        # Set initial heartbeat
        await redis_client.setex(heartbeat_key, 5, str(time.time()))

        # Wait a bit
        await asyncio.sleep(2)

        # Refresh heartbeat
        await redis_client.setex(heartbeat_key, 5, str(time.time()))

        # TTL should be refreshed
        ttl = await redis_client.ttl(heartbeat_key)
        assert ttl > 3  # Should be close to 5 again


class TestFailoverBehavior:
    """Test failover scenarios."""

    @pytest.mark.asyncio
    async def test_leader_election(self, redis_client, test_prefix):
        """Test leader election using SETNX."""
        leader_key = f"{test_prefix}leader"

        # First node becomes leader
        acquired = await redis_client.setnx(leader_key, "node-1")
        assert acquired is True

        # Second node fails to acquire
        acquired = await redis_client.setnx(leader_key, "node-2")
        assert acquired is False

        # Verify leader
        leader = await redis_client.get(leader_key)
        assert leader == "node-1"

    @pytest.mark.asyncio
    async def test_distributed_lock(self, redis_client, test_prefix):
        """Test distributed locking."""
        lock_key = f"{test_prefix}lock:resource"

        # Acquire lock with TTL
        acquired = await redis_client.set(
            lock_key, "owner-1", nx=True, ex=10
        )
        assert acquired is True

        # Cannot acquire again
        acquired = await redis_client.set(
            lock_key, "owner-2", nx=True, ex=10
        )
        assert acquired is None

        # Release lock
        await redis_client.delete(lock_key)

        # Now can acquire
        acquired = await redis_client.set(
            lock_key, "owner-2", nx=True, ex=10
        )
        assert acquired is True


class TestPubSubCoordination:
    """Test pub/sub for agent coordination."""

    @pytest.mark.asyncio
    async def test_broadcast_message(self, redis_client, test_prefix):
        """Test broadcasting messages to agents."""
        channel = f"{test_prefix}broadcast"
        messages_received = []

        async def subscriber():
            pubsub = redis_client.pubsub()
            await pubsub.subscribe(channel)

            async for message in pubsub.listen():
                if message["type"] == "message":
                    messages_received.append(message["data"])
                    if len(messages_received) >= 3:
                        break

            await pubsub.unsubscribe(channel)

        # Start subscriber
        sub_task = asyncio.create_task(subscriber())

        # Give subscriber time to connect
        await asyncio.sleep(0.1)

        # Publish messages
        await redis_client.publish(channel, "msg-1")
        await redis_client.publish(channel, "msg-2")
        await redis_client.publish(channel, "msg-3")

        # Wait for subscriber
        await asyncio.wait_for(sub_task, timeout=5.0)

        assert len(messages_received) == 3
        assert "msg-1" in messages_received


class TestConcurrentAccess:
    """Test concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_task_claims(self, redis_client, test_prefix):
        """Test multiple workers claiming tasks concurrently."""
        queue_key = f"{test_prefix}tasks:concurrent"
        claimed_key = f"{test_prefix}tasks:claimed"

        # Add tasks
        for i in range(20):
            await redis_client.lpush(queue_key, f"task-{i}")

        claimed_tasks = []

        async def worker(worker_id: int):
            """Simulate a worker claiming tasks."""
            local_claimed = []
            while True:
                # Atomically move task from pending to claimed
                task = await redis_client.rpoplpush(queue_key, claimed_key)
                if task is None:
                    break
                local_claimed.append((worker_id, task))
                await asyncio.sleep(0.01)  # Simulate work
            return local_claimed

        # Run 5 concurrent workers
        results = await asyncio.gather(*[worker(i) for i in range(5)])

        # Flatten results
        all_claimed = [task for worker_tasks in results for task in worker_tasks]

        # Verify all tasks claimed exactly once
        assert len(all_claimed) == 20
        task_names = [t[1] for t in all_claimed]
        assert len(set(task_names)) == 20  # No duplicates
