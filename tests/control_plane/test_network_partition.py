"""
Tests for Network Partition and Failure Mode Scenarios.

Tests critical multi-region failure scenarios:
- Network partition simulation
- Stale leader detection
- Event ordering guarantees
- Cross-region state consistency
- Failover during task execution
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.control_plane.regional_sync import (
    RegionalEvent,
    RegionalEventType,
    RegionalSyncConfig,
)


# =============================================================================
# Mock Infrastructure for Partition Tests
# =============================================================================


@dataclass
class MockRegionState:
    """Tracks state for a mock region during tests."""

    region_id: str
    is_leader: bool = False
    is_connected: bool = True
    received_events: List[RegionalEvent] = field(default_factory=list)
    buffered_events: List[RegionalEvent] = field(default_factory=list)
    known_entities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_leader_heartbeat: float = 0.0
    leader_id: Optional[str] = None


class MockPartitionableEventBus:
    """Mock event bus that can simulate network partitions."""

    def __init__(self, config: RegionalSyncConfig):
        self.config = config
        self.connected_regions: Set[str] = set()
        self.partition_matrix: Dict[str, Set[str]] = {}  # region -> reachable regions
        self.published_events: List[RegionalEvent] = []
        self.subscriptions: Dict[str, List[callable]] = {}

    async def connect(self, region_id: str) -> None:
        """Connect a region."""
        self.connected_regions.add(region_id)
        # New region can reach all existing regions
        self.partition_matrix[region_id] = set(self.connected_regions)
        # All existing regions can reach the new region
        for existing_region in self.connected_regions:
            if existing_region != region_id:
                self.partition_matrix[existing_region].add(region_id)

    async def disconnect(self, region_id: str) -> None:
        """Disconnect a region."""
        self.connected_regions.discard(region_id)
        if region_id in self.partition_matrix:
            del self.partition_matrix[region_id]

    def create_partition(self, region_a: str, region_b: str) -> None:
        """Create a network partition between two regions."""
        if region_a in self.partition_matrix:
            self.partition_matrix[region_a].discard(region_b)
        if region_b in self.partition_matrix:
            self.partition_matrix[region_b].discard(region_a)

    def heal_partition(self, region_a: str, region_b: str) -> None:
        """Heal a network partition between two regions."""
        if region_a in self.partition_matrix and region_a in self.connected_regions:
            self.partition_matrix[region_a].add(region_b)
        if region_b in self.partition_matrix and region_b in self.connected_regions:
            self.partition_matrix[region_b].add(region_a)

    def can_reach(self, from_region: str, to_region: str) -> bool:
        """Check if one region can reach another."""
        if from_region not in self.partition_matrix:
            return False
        return to_region in self.partition_matrix[from_region]

    async def publish(self, event: RegionalEvent, target_region: Optional[str] = None) -> int:
        """Publish event, respecting partition matrix."""
        self.published_events.append(event)
        delivered = 0

        for region_id, handlers in self.subscriptions.items():
            if region_id == event.source_region:
                continue  # Don't deliver to self

            if not self.can_reach(event.source_region, region_id):
                continue  # Partitioned

            if target_region and region_id != target_region:
                continue  # Targeted delivery

            for handler in handlers:
                try:
                    await handler(event)
                    delivered += 1
                except Exception:
                    pass

        return delivered

    async def subscribe(self, region_id: str, handler: callable) -> None:
        """Subscribe a region to events."""
        if region_id not in self.subscriptions:
            self.subscriptions[region_id] = []
        self.subscriptions[region_id].append(handler)


# =============================================================================
# Network Partition Tests
# =============================================================================


class TestNetworkPartitionSimulation:
    """Tests for simulating network partitions."""

    @pytest.fixture
    def event_bus(self):
        """Create mock partitionable event bus."""
        config = RegionalSyncConfig(local_region="us-west-2")
        return MockPartitionableEventBus(config)

    @pytest.fixture
    def regions(self):
        """Create mock region states."""
        return {
            "us-west-2": MockRegionState(region_id="us-west-2", is_leader=True),
            "us-east-1": MockRegionState(region_id="us-east-1"),
            "eu-west-1": MockRegionState(region_id="eu-west-1"),
        }

    @pytest.mark.asyncio
    async def test_partition_blocks_events(self, event_bus, regions):
        """Test that partitioned regions don't receive events."""
        # Connect all regions
        for region_id in regions:
            await event_bus.connect(region_id)

        # Set up event handlers - use factory function to capture state properly
        def make_handler(s):
            async def handler(event):
                s.received_events.append(event)

            return handler

        for region_id, state in regions.items():
            await event_bus.subscribe(region_id, make_handler(state))

        # Publish event from us-west-2
        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_REGISTERED,
            source_region="us-west-2",
            entity_id="agent-001",
        )
        delivered = await event_bus.publish(event)
        assert delivered == 2  # Both us-east-1 and eu-west-1

        # Create partition between us-west-2 and eu-west-1
        event_bus.create_partition("us-west-2", "eu-west-1")

        # Publish another event
        event2 = RegionalEvent(
            event_type=RegionalEventType.AGENT_UPDATED,
            source_region="us-west-2",
            entity_id="agent-001",
        )
        delivered2 = await event_bus.publish(event2)
        assert delivered2 == 1  # Only us-east-1

        # Verify eu-west-1 didn't receive second event
        assert len(regions["us-east-1"].received_events) == 2
        assert len(regions["eu-west-1"].received_events) == 1

    @pytest.mark.asyncio
    async def test_partition_heal_restores_connectivity(self, event_bus, regions):
        """Test that healing partition restores event delivery."""
        # Connect all regions
        for region_id in regions:
            await event_bus.connect(region_id)

        def make_handler(s):
            async def handler(event):
                s.received_events.append(event)

            return handler

        for region_id, state in regions.items():
            await event_bus.subscribe(region_id, make_handler(state))

        # Create and heal partition
        event_bus.create_partition("us-west-2", "eu-west-1")
        event_bus.heal_partition("us-west-2", "eu-west-1")

        # Publish event
        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_REGISTERED,
            source_region="us-west-2",
            entity_id="agent-002",
        )
        delivered = await event_bus.publish(event)
        assert delivered == 2  # Both regions reachable again

    @pytest.mark.asyncio
    async def test_asymmetric_partition(self, event_bus, regions):
        """Test asymmetric partition where A→B works but B→A fails."""
        for region_id in regions:
            await event_bus.connect(region_id)

        def make_handler(s):
            async def handler(event):
                s.received_events.append(event)

            return handler

        for region_id, state in regions.items():
            await event_bus.subscribe(region_id, make_handler(state))

        # Create asymmetric partition: us-west-2 can reach us-east-1 but not vice versa
        event_bus.partition_matrix["us-east-1"].discard("us-west-2")

        # Event from us-west-2 reaches us-east-1
        event1 = RegionalEvent(
            event_type=RegionalEventType.AGENT_REGISTERED,
            source_region="us-west-2",
            entity_id="agent-003",
        )
        await event_bus.publish(event1)
        assert len(regions["us-east-1"].received_events) == 1

        # Event from us-east-1 doesn't reach us-west-2
        event2 = RegionalEvent(
            event_type=RegionalEventType.AGENT_REGISTERED,
            source_region="us-east-1",
            entity_id="agent-004",
        )
        await event_bus.publish(event2)
        assert len(regions["us-west-2"].received_events) == 0

    @pytest.mark.asyncio
    async def test_total_partition_isolation(self, event_bus, regions):
        """Test that fully partitioned region is completely isolated."""
        for region_id in regions:
            await event_bus.connect(region_id)

        def make_handler(s):
            async def handler(event):
                s.received_events.append(event)

            return handler

        for region_id, state in regions.items():
            await event_bus.subscribe(region_id, make_handler(state))

        # Partition eu-west-1 from everyone
        event_bus.create_partition("us-west-2", "eu-west-1")
        event_bus.create_partition("us-east-1", "eu-west-1")

        # Events from us-west-2 only reach us-east-1
        event = RegionalEvent(
            event_type=RegionalEventType.LEADER_ELECTED,
            source_region="us-west-2",
            entity_id="leader",
        )
        await event_bus.publish(event)

        assert len(regions["us-east-1"].received_events) == 1
        assert len(regions["eu-west-1"].received_events) == 0


# =============================================================================
# Stale Leader Detection Tests
# =============================================================================


class TestStaleLeaderDetection:
    """Tests for detecting and handling stale leaders."""

    def test_leader_heartbeat_timeout(self):
        """Test that missing heartbeats trigger leader timeout."""
        state = MockRegionState(region_id="us-west-2")
        state.last_leader_heartbeat = time.time() - 35  # 35 seconds ago

        timeout = 30.0
        is_stale = (time.time() - state.last_leader_heartbeat) > timeout
        assert is_stale is True

    def test_fresh_leader_heartbeat(self):
        """Test that recent heartbeat prevents timeout."""
        state = MockRegionState(region_id="us-west-2")
        state.last_leader_heartbeat = time.time() - 5  # 5 seconds ago

        timeout = 30.0
        is_stale = (time.time() - state.last_leader_heartbeat) > timeout
        assert is_stale is False

    @pytest.mark.asyncio
    async def test_leader_change_on_stale_detection(self):
        """Test that stale leader triggers re-election."""
        regions = {
            "us-west-2": MockRegionState(region_id="us-west-2", is_leader=True),
            "us-east-1": MockRegionState(region_id="us-east-1"),
        }

        # Simulate stale leader detection
        regions["us-west-2"].last_leader_heartbeat = time.time() - 60
        timeout = 30.0

        # Check from us-east-1's perspective
        is_stale = (time.time() - regions["us-west-2"].last_leader_heartbeat) > timeout

        if is_stale:
            # Trigger re-election
            regions["us-west-2"].is_leader = False
            regions["us-east-1"].is_leader = True

        assert regions["us-west-2"].is_leader is False
        assert regions["us-east-1"].is_leader is True


# =============================================================================
# Event Ordering Tests
# =============================================================================


class TestEventOrdering:
    """Tests for event ordering guarantees."""

    def test_newer_event_wins_conflict(self):
        """Test that newer timestamp wins in conflict resolution."""
        older_event = RegionalEvent(
            event_type=RegionalEventType.AGENT_UPDATED,
            source_region="us-west-2",
            entity_id="agent-001",
            timestamp=1000.0,
            data={"status": "healthy"},
        )

        newer_event = RegionalEvent(
            event_type=RegionalEventType.AGENT_UPDATED,
            source_region="us-east-1",
            entity_id="agent-001",
            timestamp=1001.0,
            data={"status": "unhealthy"},
        )

        assert newer_event.is_newer_than(older_event) is True
        assert older_event.is_newer_than(newer_event) is False

    def test_same_timestamp_deterministic(self):
        """Test that same-timestamp events resolve deterministically."""
        event1 = RegionalEvent(
            event_type=RegionalEventType.AGENT_UPDATED,
            source_region="us-west-2",
            entity_id="agent-001",
            timestamp=1000.0,
        )

        event2 = RegionalEvent(
            event_type=RegionalEventType.AGENT_UPDATED,
            source_region="us-east-1",
            entity_id="agent-001",
            timestamp=1000.0,
        )

        # Neither is newer
        assert event1.is_newer_than(event2) is False
        assert event2.is_newer_than(event1) is False

        # Use source_region as tiebreaker (lexicographic)
        winner = event1 if event1.source_region < event2.source_region else event2
        assert winner.source_region == "us-east-1"  # 'us-east-1' < 'us-west-2'

    def test_out_of_order_event_ignored(self):
        """Test that out-of-order events are properly ignored."""
        state = {"agent-001": {"status": "healthy", "timestamp": 1001.0}}

        # Old event arrives late
        old_event = RegionalEvent(
            event_type=RegionalEventType.AGENT_UPDATED,
            source_region="us-west-2",
            entity_id="agent-001",
            timestamp=1000.0,
            data={"status": "unhealthy"},
        )

        # Apply only if newer
        if old_event.timestamp > state["agent-001"]["timestamp"]:
            state["agent-001"]["status"] = old_event.data["status"]

        # State unchanged - old event ignored
        assert state["agent-001"]["status"] == "healthy"


# =============================================================================
# Cross-Region State Consistency Tests
# =============================================================================


class TestCrossRegionConsistency:
    """Tests for cross-region state consistency."""

    def test_state_convergence_after_events(self):
        """Test that states converge after applying events in order."""
        regions = {
            "us-west-2": {"agent-001": None},
            "us-east-1": {"agent-001": None},
        }

        events = [
            RegionalEvent(
                event_type=RegionalEventType.AGENT_REGISTERED,
                source_region="us-west-2",
                entity_id="agent-001",
                timestamp=1000.0,
                data={"name": "Claude", "version": 1},
            ),
            RegionalEvent(
                event_type=RegionalEventType.AGENT_UPDATED,
                source_region="us-east-1",
                entity_id="agent-001",
                timestamp=1001.0,
                data={"name": "Claude", "version": 2},
            ),
        ]

        # Apply events to both regions
        for event in events:
            for region_state in regions.values():
                current = region_state.get(event.entity_id) or {}
                current_ts = current.get("timestamp", 0)
                if event.timestamp > current_ts:
                    region_state[event.entity_id] = {
                        **event.data,
                        "timestamp": event.timestamp,
                    }

        # States should be identical
        assert regions["us-west-2"]["agent-001"]["version"] == 2
        assert regions["us-east-1"]["agent-001"]["version"] == 2

    def test_partition_causes_divergence(self):
        """Test that partition causes state divergence."""
        regions = {
            "us-west-2": {"agent-001": {"status": "healthy", "timestamp": 1000.0}},
            "us-east-1": {"agent-001": {"status": "healthy", "timestamp": 1000.0}},
        }

        # During partition, us-west-2 updates but us-east-1 doesn't receive
        regions["us-west-2"]["agent-001"] = {"status": "unhealthy", "timestamp": 1001.0}

        # States diverged
        assert regions["us-west-2"]["agent-001"]["status"] == "unhealthy"
        assert regions["us-east-1"]["agent-001"]["status"] == "healthy"

    def test_partition_heal_converges(self):
        """Test that healing partition allows convergence."""
        regions = {
            "us-west-2": {"agent-001": {"status": "unhealthy", "timestamp": 1001.0}},
            "us-east-1": {"agent-001": {"status": "healthy", "timestamp": 1000.0}},
        }

        # After partition heals, sync event sent
        sync_event = RegionalEvent(
            event_type=RegionalEventType.AGENT_UPDATED,
            source_region="us-west-2",
            entity_id="agent-001",
            timestamp=1001.0,
            data={"status": "unhealthy"},
        )

        # Apply to us-east-1
        current_ts = regions["us-east-1"]["agent-001"]["timestamp"]
        if sync_event.timestamp > current_ts:
            regions["us-east-1"]["agent-001"] = {
                "status": sync_event.data["status"],
                "timestamp": sync_event.timestamp,
            }

        # States converged
        assert regions["us-west-2"]["agent-001"]["status"] == "unhealthy"
        assert regions["us-east-1"]["agent-001"]["status"] == "unhealthy"


# =============================================================================
# Failover During Task Execution Tests
# =============================================================================


class TestFailoverDuringTaskExecution:
    """Tests for failover scenarios during task execution."""

    @pytest.fixture
    def task_state(self):
        """Create mock task state."""
        return {
            "task-001": {
                "id": "task-001",
                "status": "running",
                "assigned_region": "us-west-2",
                "assigned_agent": "agent-001",
                "started_at": time.time() - 30,
                "timeout": 60.0,
            }
        }

    def test_task_failover_on_region_failure(self, task_state):
        """Test task is reassigned when region fails."""
        task = task_state["task-001"]

        # Simulate region failure
        failed_region = task["assigned_region"]
        available_regions = ["us-east-1", "eu-west-1"]

        # Failover logic
        if failed_region not in available_regions:
            task["status"] = "pending"  # Return to queue
            task["assigned_region"] = None
            task["assigned_agent"] = None
            task["failover_count"] = task.get("failover_count", 0) + 1

        assert task["status"] == "pending"
        assert task["failover_count"] == 1

    def test_task_timeout_during_failover(self, task_state):
        """Test task timeout is handled during failover."""
        task = task_state["task-001"]
        # Set started_at to a fixed time that's definitely past timeout
        current_time = time.time()
        task["started_at"] = current_time - 70  # 70 seconds ago
        task["timeout"] = 60.0

        elapsed = current_time - task["started_at"]  # Use same reference time
        is_timed_out = elapsed > task["timeout"]

        if is_timed_out:
            task["status"] = "failed"
            task["error"] = "Task timed out during region failover"

        assert task["status"] == "failed"
        assert "timed out" in task["error"]

    def test_task_state_preserved_during_failover(self, task_state):
        """Test task intermediate state is preserved during failover."""
        task = task_state["task-001"]
        task["intermediate_result"] = {"progress": 50, "partial_data": "some_data"}

        # Simulate failover
        original_result = task.get("intermediate_result")

        task["status"] = "pending"
        task["assigned_region"] = "us-east-1"
        task["resumed_at"] = time.time()
        # Preserve intermediate result
        task["intermediate_result"] = original_result

        assert task["intermediate_result"]["progress"] == 50
        assert task["resumed_at"] is not None

    def test_multiple_failover_limit(self, task_state):
        """Test task fails after max failover attempts."""
        task = task_state["task-001"]
        max_failovers = 3

        # Simulate multiple failovers
        for _ in range(4):
            task["failover_count"] = task.get("failover_count", 0) + 1
            if task["failover_count"] >= max_failovers:
                task["status"] = "failed"
                task["error"] = f"Max failover attempts ({max_failovers}) exceeded"
                break

        assert task["status"] == "failed"
        assert "Max failover" in task["error"]


# =============================================================================
# Buffer Exhaustion Tests
# =============================================================================


class TestEventBufferExhaustion:
    """Tests for event buffer exhaustion scenarios."""

    def test_buffer_overflow_drops_oldest(self):
        """Test that buffer overflow drops oldest events."""
        max_buffer = 5
        buffer: List[RegionalEvent] = []

        # Fill buffer beyond capacity
        for i in range(10):
            event = RegionalEvent(
                event_type=RegionalEventType.AGENT_HEARTBEAT,
                source_region="us-west-2",
                entity_id=f"agent-{i:03d}",
                timestamp=float(i),
            )
            buffer.append(event)
            if len(buffer) > max_buffer:
                buffer.pop(0)  # Drop oldest

        assert len(buffer) == max_buffer
        assert buffer[0].entity_id == "agent-005"  # Oldest remaining

    def test_buffer_metrics_tracked(self):
        """Test that buffer overflow is tracked in metrics."""
        metrics = {"events_buffered": 0, "events_dropped": 0}
        max_buffer = 3
        buffer: List[RegionalEvent] = []

        for i in range(5):
            event = RegionalEvent(
                event_type=RegionalEventType.AGENT_HEARTBEAT,
                source_region="us-west-2",
                entity_id=f"agent-{i:03d}",
            )
            if len(buffer) >= max_buffer:
                buffer.pop(0)
                metrics["events_dropped"] += 1
            buffer.append(event)
            metrics["events_buffered"] += 1

        assert metrics["events_buffered"] == 5
        assert metrics["events_dropped"] == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestPartitionRecoveryIntegration:
    """Integration tests for partition recovery scenarios."""

    @pytest.mark.asyncio
    async def test_full_partition_recovery_cycle(self):
        """Test complete partition → divergence → heal → convergence cycle."""
        config = RegionalSyncConfig(local_region="us-west-2")
        event_bus = MockPartitionableEventBus(config)

        regions = {
            "us-west-2": MockRegionState(region_id="us-west-2"),
            "us-east-1": MockRegionState(region_id="us-east-1"),
        }

        # Connect regions
        for region_id in regions:
            await event_bus.connect(region_id)

        def make_handler(s):
            async def handler(event):
                s.received_events.append(event)
                s.known_entities[event.entity_id] = event.data

            return handler

        for region_id, state in regions.items():
            await event_bus.subscribe(region_id, make_handler(state))

        # 1. Normal operation - both receive events
        event1 = RegionalEvent(
            event_type=RegionalEventType.AGENT_REGISTERED,
            source_region="us-west-2",
            entity_id="agent-001",
            data={"status": "v1"},
        )
        await event_bus.publish(event1)
        assert len(regions["us-east-1"].received_events) == 1

        # 2. Create partition
        event_bus.create_partition("us-west-2", "us-east-1")

        # 3. Updates during partition
        event2 = RegionalEvent(
            event_type=RegionalEventType.AGENT_UPDATED,
            source_region="us-west-2",
            entity_id="agent-001",
            data={"status": "v2"},
        )
        await event_bus.publish(event2)

        # us-east-1 still has v1
        assert regions["us-east-1"].known_entities.get("agent-001", {}).get("status") == "v1"

        # 4. Heal partition
        event_bus.heal_partition("us-west-2", "us-east-1")

        # 5. Sync event restores consistency
        sync_event = RegionalEvent(
            event_type=RegionalEventType.AGENT_UPDATED,
            source_region="us-west-2",
            entity_id="agent-001",
            data={"status": "v2"},
        )
        await event_bus.publish(sync_event)

        # us-east-1 now has v2
        assert regions["us-east-1"].known_entities.get("agent-001", {}).get("status") == "v2"
