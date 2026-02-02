"""
Extended tests for TaskScheduler covering policy enforcement, cost constraints,
Redis operations, dead-letter handling, and stale task recovery.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.control_plane.scheduler import (
    Task,
    TaskPriority,
    TaskScheduler,
    TaskStatus,
    RegionRoutingMode,
)


class MockPolicyError(Exception):
    """Mock policy violation error."""

    def __init__(self, result=None, task_type=None, region=None, **kwargs):
        self.result = result
        self.task_type = task_type
        self.region = region
        super().__init__(f"Policy violation: {task_type}")


class MockCostError(Exception):
    """Mock cost limit exceeded error."""

    def __init__(self, result=None, task_type=None, **kwargs):
        self.result = result
        self.task_type = task_type
        super().__init__(f"Cost limit exceeded: {task_type}")


class TestPolicyEnforcement:
    """Tests for policy enforcement on submit and claim."""

    @pytest.fixture
    def mock_policy_manager(self):
        """Create a mock policy manager."""
        manager = MagicMock()
        return manager

    @pytest.fixture
    def scheduler_with_policy(self, mock_policy_manager):
        """Create scheduler with policy manager."""
        return TaskScheduler(
            redis_url="memory://",
            policy_manager=mock_policy_manager,
        )

    @pytest.mark.asyncio
    async def test_submit_hard_policy_violation_raises(self, scheduler_with_policy, mock_policy_manager):
        """Test that HARD policy violation on submit raises error."""
        with patch("aragora.control_plane.scheduler.HAS_POLICY", True):
            # Mock policy module imports
            mock_enforcement = MagicMock()
            mock_enforcement.HARD = "hard"
            mock_enforcement.WARN = "warn"

            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.enforcement_level = mock_enforcement.HARD
            mock_result.reason = "Task type not allowed"
            mock_result.policy_id = "policy-123"

            mock_policy_manager.evaluate_task_dispatch.return_value = mock_result

            with (
                patch("aragora.control_plane.scheduler.EnforcementLevel", mock_enforcement),
                patch("aragora.control_plane.scheduler.PolicyViolationError", MockPolicyError),
            ):
                with pytest.raises(MockPolicyError):
                    await scheduler_with_policy.submit(
                        task_type="prohibited",
                        payload={},
                    )

    @pytest.mark.asyncio
    async def test_submit_warn_policy_violation_logs_but_allows(
        self, scheduler_with_policy, mock_policy_manager
    ):
        """Test that WARN policy violation logs warning but allows submission."""
        with patch("aragora.control_plane.scheduler.HAS_POLICY", True):
            mock_enforcement = MagicMock()
            mock_enforcement.HARD = "hard"
            mock_enforcement.WARN = "warn"

            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.enforcement_level = mock_enforcement.WARN
            mock_result.reason = "Not recommended"
            mock_result.policy_id = "policy-456"

            mock_policy_manager.evaluate_task_dispatch.return_value = mock_result

            with patch("aragora.control_plane.scheduler.EnforcementLevel", mock_enforcement):
                task_id = await scheduler_with_policy.submit(
                    task_type="warned",
                    payload={},
                )

            assert task_id is not None
            task = await scheduler_with_policy.get(task_id)
            assert task is not None

    @pytest.mark.asyncio
    async def test_submit_allowed_by_policy(self, scheduler_with_policy, mock_policy_manager):
        """Test successful submission when policy allows."""
        with patch("aragora.control_plane.scheduler.HAS_POLICY", True):
            mock_result = MagicMock()
            mock_result.allowed = True

            mock_policy_manager.evaluate_task_dispatch.return_value = mock_result

            task_id = await scheduler_with_policy.submit(
                task_type="allowed",
                payload={},
            )

            assert task_id is not None

    @pytest.mark.asyncio
    async def test_claim_policy_rejection_requeues_task(
        self, scheduler_with_policy, mock_policy_manager
    ):
        """Test that claim policy rejection requeues task for another worker."""
        with patch("aragora.control_plane.scheduler.HAS_POLICY", True):
            mock_enforcement = MagicMock()
            mock_enforcement.HARD = "hard"
            mock_enforcement.WARN = "warn"

            # Allow submission
            submit_result = MagicMock()
            submit_result.allowed = True

            # Reject claim
            claim_result = MagicMock()
            claim_result.allowed = False
            claim_result.enforcement_level = mock_enforcement.HARD
            claim_result.reason = "Worker not authorized"
            claim_result.policy_id = "policy-789"

            mock_policy_manager.evaluate_task_dispatch.side_effect = [
                submit_result,  # For submit
                claim_result,   # For first claim
            ]

            with patch("aragora.control_plane.scheduler.EnforcementLevel", mock_enforcement):
                task_id = await scheduler_with_policy.submit(
                    task_type="restricted",
                    payload={},
                )

                # First worker should be rejected
                task = await scheduler_with_policy.claim(
                    worker_id="unauthorized-worker",
                    capabilities=[],
                )

                # Task should be rejected (None returned)
                assert task is None

                # Check metadata was updated
                stored_task = await scheduler_with_policy.get(task_id)
                assert stored_task.metadata.get("policy_rejection_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_claim_policy_warn_logs_and_skips(
        self, scheduler_with_policy, mock_policy_manager
    ):
        """Test that claim WARN policy rejection logs and skips task."""
        with patch("aragora.control_plane.scheduler.HAS_POLICY", True):
            mock_enforcement = MagicMock()
            mock_enforcement.HARD = "hard"
            mock_enforcement.WARN = "warn"

            # Allow submission
            submit_result = MagicMock()
            submit_result.allowed = True

            # Warn on claim
            claim_result = MagicMock()
            claim_result.allowed = False
            claim_result.enforcement_level = mock_enforcement.WARN
            claim_result.reason = "Not recommended for this worker"
            claim_result.policy_id = "policy-warn"

            mock_policy_manager.evaluate_task_dispatch.side_effect = [
                submit_result,
                claim_result,
            ]

            with patch("aragora.control_plane.scheduler.EnforcementLevel", mock_enforcement):
                await scheduler_with_policy.submit(
                    task_type="warned-claim",
                    payload={},
                )

                task = await scheduler_with_policy.claim(
                    worker_id="warned-worker",
                    capabilities=[],
                )

                # Task should be rejected
                assert task is None


class TestCostEnforcement:
    """Tests for cost constraint enforcement."""

    @pytest.fixture
    def mock_cost_enforcer(self):
        """Create a mock cost enforcer."""
        enforcer = MagicMock()
        return enforcer

    @pytest.fixture
    def scheduler_with_cost(self, mock_cost_enforcer):
        """Create scheduler with cost enforcer."""
        return TaskScheduler(
            redis_url="memory://",
            cost_enforcer=mock_cost_enforcer,
        )

    @pytest.mark.asyncio
    async def test_submit_cost_limit_exceeded_raises(
        self, scheduler_with_cost, mock_cost_enforcer
    ):
        """Test that exceeding cost limit raises error."""
        with patch("aragora.control_plane.scheduler.HAS_COST_ENFORCEMENT", True):
            mock_result = MagicMock()
            mock_result.allowed = False
            mock_result.budget_percentage_used = 105.0
            mock_result.throttle_level = MagicMock(value="blocked")

            mock_cost_enforcer.check_budget_constraint.return_value = mock_result

            with patch(
                "aragora.control_plane.scheduler.CostLimitExceededError",
                MockCostError,
            ):
                with pytest.raises(MockCostError):
                    await scheduler_with_cost.submit(
                        task_type="expensive",
                        payload={},
                        workspace_id="over-budget-ws",
                    )

    @pytest.mark.asyncio
    async def test_submit_priority_throttled(self, scheduler_with_cost, mock_cost_enforcer):
        """Test that priority is throttled when approaching budget limit."""
        with patch("aragora.control_plane.scheduler.HAS_COST_ENFORCEMENT", True):
            mock_result = MagicMock()
            mock_result.allowed = True
            mock_result.priority_adjustment = -2  # Decrease priority
            mock_result.throttle_level = MagicMock(value="medium")
            mock_result.budget_percentage_used = 85.0

            mock_cost_enforcer.check_budget_constraint.return_value = mock_result

            task_id = await scheduler_with_cost.submit(
                task_type="throttled",
                payload={},
                priority=TaskPriority.HIGH,
                workspace_id="throttled-ws",
            )

            task = await scheduler_with_cost.get(task_id)
            # Priority should be lowered
            assert task.priority.value < TaskPriority.HIGH.value
            # Cost constraint info should be in metadata
            assert "cost_constraint" in task.metadata
            assert task.metadata["cost_constraint"]["original_priority"] == "HIGH"

    @pytest.mark.asyncio
    async def test_submit_no_throttling_within_budget(
        self, scheduler_with_cost, mock_cost_enforcer
    ):
        """Test normal submission when within budget."""
        with patch("aragora.control_plane.scheduler.HAS_COST_ENFORCEMENT", True):
            mock_result = MagicMock()
            mock_result.allowed = True
            mock_result.priority_adjustment = 0
            mock_result.throttle_level = MagicMock(value="none")
            mock_result.budget_percentage_used = 30.0

            mock_cost_enforcer.check_budget_constraint.return_value = mock_result

            task_id = await scheduler_with_cost.submit(
                task_type="normal",
                payload={},
                priority=TaskPriority.HIGH,
            )

            task = await scheduler_with_cost.get(task_id)
            assert task.priority == TaskPriority.HIGH

    def test_set_cost_enforcer(self):
        """Test setting cost enforcer after creation."""
        scheduler = TaskScheduler(redis_url="memory://")
        assert scheduler._cost_enforcer is None

        mock_enforcer = MagicMock()
        scheduler.set_cost_enforcer(mock_enforcer)

        assert scheduler._cost_enforcer == mock_enforcer

    def test_get_budget_status_no_enforcer(self):
        """Test getting budget status without enforcer."""
        scheduler = TaskScheduler(redis_url="memory://")
        status = scheduler.get_budget_status(workspace_id="test")

        assert "error" in status

    def test_get_budget_status_with_enforcer(self):
        """Test getting budget status with enforcer."""
        mock_enforcer = MagicMock()
        mock_enforcer.get_budget_status.return_value = {
            "budget_used": 50.0,
            "budget_limit": 100.0,
            "remaining": 50.0,
        }

        scheduler = TaskScheduler(redis_url="memory://", cost_enforcer=mock_enforcer)
        status = scheduler.get_budget_status(workspace_id="test")

        assert status["budget_used"] == 50.0
        mock_enforcer.get_budget_status.assert_called_once()


def create_mock_redis():
    """Create a properly configured mock Redis client."""
    redis = AsyncMock()
    redis.ping = AsyncMock()
    redis.xgroup_create = AsyncMock()
    redis.get = AsyncMock()
    redis.set = AsyncMock()
    redis.xadd = AsyncMock()
    redis.xreadgroup = AsyncMock()
    redis.xack = AsyncMock()
    redis.xdel = AsyncMock()
    redis.sadd = AsyncMock()
    redis.srem = AsyncMock()
    redis.smembers = AsyncMock()
    redis.expire = AsyncMock()
    redis.close = AsyncMock()
    redis.scan_iter = AsyncMock()
    redis.xpending_range = AsyncMock()
    redis.xclaim = AsyncMock()

    # Mock pipeline as async context manager
    pipeline_mock = AsyncMock()
    pipeline_mock.srem = MagicMock()
    pipeline_mock.sadd = MagicMock()
    pipeline_mock.expire = MagicMock()
    pipeline_mock.execute = AsyncMock()

    # Create async context manager
    async def async_pipeline_cm():
        return pipeline_mock

    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=pipeline_mock)
    cm.__aexit__ = AsyncMock(return_value=None)
    redis.pipeline = MagicMock(return_value=cm)

    return redis


class TestRedisOperations:
    """Tests for Redis-specific operations with mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return create_mock_redis()

    @pytest.mark.asyncio
    async def test_connect_redis_success(self, mock_redis):
        """Test successful Redis connection."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            scheduler = TaskScheduler(redis_url="redis://localhost:6379")
            await scheduler.connect()

            mock_redis.ping.assert_called_once()
            assert scheduler._redis is not None

    @pytest.mark.asyncio
    async def test_connect_redis_failure_falls_back(self):
        """Test Redis connection failure falls back to in-memory."""
        with patch("redis.asyncio.from_url", side_effect=ConnectionError("Connection refused")):
            scheduler = TaskScheduler(redis_url="redis://localhost:6379")
            await scheduler.connect()

            # Should fall back to in-memory
            assert scheduler._redis is None

    @pytest.mark.asyncio
    async def test_connect_memory_url(self):
        """Test explicit memory:// URL uses in-memory."""
        scheduler = TaskScheduler(redis_url="memory://")
        await scheduler.connect()

        assert scheduler._redis is None

    @pytest.mark.asyncio
    async def test_claim_from_redis_capability_mismatch(self, mock_redis):
        """Test Redis claim requeues task on capability mismatch."""
        # Setup task in Redis
        task_data = Task(
            id="task-123",
            task_type="code",
            payload={},
            required_capabilities={"python", "testing"},
        ).to_dict()

        mock_redis.xreadgroup.return_value = [
            ("stream-key", [("msg-1", {"task_id": "task-123", "task_type": "code"})])
        ]
        mock_redis.get.return_value = __import__("json").dumps(task_data)

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            scheduler = TaskScheduler(redis_url="redis://localhost:6379")
            scheduler._redis = mock_redis

            # Claim with missing capability
            task = await scheduler.claim(
                worker_id="worker-1",
                capabilities=["python"],  # Missing "testing"
            )

            # Should return None and requeue
            assert task is None
            # Should have called xack to release from this worker
            mock_redis.xack.assert_called()

    @pytest.mark.asyncio
    async def test_claim_from_redis_deleted_task(self, mock_redis):
        """Test Redis claim handles deleted task gracefully."""
        mock_redis.xreadgroup.return_value = [
            ("stream-key", [("msg-1", {"task_id": "deleted-task", "task_type": "test"})])
        ]
        mock_redis.get.return_value = None  # Task was deleted

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            scheduler = TaskScheduler(redis_url="redis://localhost:6379")
            scheduler._redis = mock_redis

            task = await scheduler.claim(
                worker_id="worker-1",
                capabilities=[],
            )

            assert task is None
            # Should ack and delete the orphaned message
            mock_redis.xack.assert_called()
            mock_redis.xdel.assert_called()

    @pytest.mark.asyncio
    async def test_claim_from_redis_success(self, mock_redis):
        """Test successful Redis claim."""
        task_data = Task(
            id="task-123",
            task_type="code",
            payload={"data": "test"},
            required_capabilities=set(),
        ).to_dict()

        mock_redis.xreadgroup.return_value = [
            ("stream-key", [("msg-1", {"task_id": "task-123", "task_type": "code"})])
        ]
        mock_redis.get.return_value = __import__("json").dumps(task_data)

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            scheduler = TaskScheduler(redis_url="redis://localhost:6379")
            scheduler._redis = mock_redis

            task = await scheduler.claim(
                worker_id="worker-1",
                capabilities=[],
            )

            assert task is not None
            assert task.id == "task-123"
            assert task.status == TaskStatus.RUNNING
            assert task.assigned_agent == "worker-1"
            assert "_stream_message_id" in task.metadata


class TestDeadLetterQueue:
    """Tests for dead-letter queue handling."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return create_mock_redis()

    @pytest.mark.asyncio
    async def test_fail_moves_to_dead_letter_after_retries(self, mock_redis):
        """Test task moves to dead-letter queue when retries exhausted."""
        scheduler = TaskScheduler(redis_url="redis://localhost:6379")
        scheduler._redis = mock_redis

        # Create task with no retries left
        task = Task(
            id="task-dlq",
            task_type="failing",
            payload={},
            max_retries=0,
        )
        scheduler._local_tasks[task.id] = task

        # Mock get to return the task from local storage
        async def mock_get(task_id):
            return scheduler._local_tasks.get(task_id)

        with patch.object(scheduler, "get", mock_get):
            await scheduler.fail(task.id, error="Permanent failure", requeue=True)

        # Task should be in dead-letter queue
        mock_redis.xadd.assert_called()
        call_args = mock_redis.xadd.call_args
        assert "dead_letter" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_dead_letter_includes_task_metadata(self, mock_redis):
        """Test dead-letter entry includes relevant task metadata."""
        scheduler = TaskScheduler(redis_url="redis://localhost:6379")
        scheduler._redis = mock_redis

        task = Task(
            id="task-meta",
            task_type="important",
            payload={},
            priority=TaskPriority.HIGH,
            max_retries=0,
        )
        scheduler._local_tasks[task.id] = task

        async def mock_get(task_id):
            return scheduler._local_tasks.get(task_id)

        with patch.object(scheduler, "get", mock_get):
            await scheduler.fail(task.id, error="Critical error", requeue=False)

        call_args = mock_redis.xadd.call_args
        dlq_data = call_args[0][1]
        assert dlq_data["task_id"] == "task-meta"
        assert dlq_data["task_type"] == "important"
        assert dlq_data["reason"] == "Critical error"
        assert dlq_data["original_priority"] == "HIGH"


class TestStaleTasks:
    """Tests for stale task recovery."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return create_mock_redis()

    @pytest.mark.asyncio
    async def test_claim_stale_tasks_no_redis(self):
        """Test stale task recovery returns 0 without Redis."""
        scheduler = TaskScheduler(redis_url="memory://")

        reclaimed = await scheduler.claim_stale_tasks()

        assert reclaimed == 0

    @pytest.mark.asyncio
    async def test_claim_stale_tasks_reclaims_old_tasks(self, mock_redis):
        """Test stale tasks are reclaimed from dead workers."""
        task_data = Task(
            id="stale-task",
            task_type="test",
            payload={},
            max_retries=3,
        )
        task_data.status = TaskStatus.RUNNING

        # Only return pending entry for NORMAL priority
        def xpending_side_effect(stream_key, *args, **kwargs):
            if "normal" in stream_key:
                return [
                    {
                        "message_id": "msg-1",
                        "consumer": "dead-worker",
                        "time_since_delivered": 120000,  # 2 minutes
                        "times_delivered": 1,
                    }
                ]
            return []

        mock_redis.xpending_range.side_effect = xpending_side_effect
        mock_redis.xclaim.return_value = [
            ("msg-1", {"task_id": "stale-task"})
        ]
        mock_redis.get.return_value = __import__("json").dumps(task_data.to_dict())

        scheduler = TaskScheduler(redis_url="redis://localhost:6379")
        scheduler._redis = mock_redis

        reclaimed = await scheduler.claim_stale_tasks(idle_ms=60000)

        assert reclaimed == 1
        mock_redis.xclaim.assert_called()

    @pytest.mark.asyncio
    async def test_claim_stale_tasks_handles_connection_error(self, mock_redis):
        """Test stale recovery handles Redis connection errors gracefully."""
        mock_redis.xpending_range.side_effect = ConnectionError("Redis unavailable")

        scheduler = TaskScheduler(redis_url="redis://localhost:6379")
        scheduler._redis = mock_redis

        # Should not raise, just return 0
        reclaimed = await scheduler.claim_stale_tasks()

        assert reclaimed == 0


class TestStatusIndex:
    """Tests for status index management."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return create_mock_redis()

    @pytest.mark.asyncio
    async def test_save_task_adds_to_status_index(self, mock_redis):
        """Test saving new task adds to status index."""
        scheduler = TaskScheduler(redis_url="redis://localhost:6379")
        scheduler._redis = mock_redis

        task = Task(
            id="new-task",
            task_type="test",
            payload={},
        )

        await scheduler._save_task(task)

        # Should add to status index
        mock_redis.sadd.assert_called()
        call_args = mock_redis.sadd.call_args
        assert "status:pending" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_save_task_updates_status_index_on_change(self, mock_redis):
        """Test saving task updates status index when status changes."""
        scheduler = TaskScheduler(redis_url="redis://localhost:6379")
        scheduler._redis = mock_redis

        task = Task(
            id="changing-task",
            task_type="test",
            payload={},
        )
        task.status = TaskStatus.RUNNING

        await scheduler._save_task(task, previous_status=TaskStatus.PENDING)

        # Should have used pipeline to atomically update indexes
        # Pipeline is entered via context manager
        mock_redis.pipeline.assert_called()

    @pytest.mark.asyncio
    async def test_list_by_status_uses_index(self, mock_redis):
        """Test list_by_status uses secondary index."""
        mock_redis.smembers.return_value = {"task-1", "task-2"}

        task1_data = Task(id="task-1", task_type="t1", payload={}).to_dict()
        task2_data = Task(id="task-2", task_type="t2", payload={}).to_dict()

        mock_redis.get.side_effect = [
            __import__("json").dumps(task1_data),
            __import__("json").dumps(task2_data),
        ]

        scheduler = TaskScheduler(redis_url="redis://localhost:6379")
        scheduler._redis = mock_redis

        tasks = await scheduler.list_by_status(TaskStatus.PENDING)

        assert len(tasks) == 2
        mock_redis.smembers.assert_called_once()

    @pytest.mark.asyncio
    async def test_local_status_index_updated(self):
        """Test local status index is updated on task changes."""
        scheduler = TaskScheduler(redis_url="memory://")

        task_id = await scheduler.submit(task_type="test", payload={})

        # Check local index has the task
        assert task_id in scheduler._local_status_index[TaskStatus.PENDING]

        # Claim the task
        await scheduler.claim(worker_id="w1", capabilities=[])

        # Should have moved in local index
        assert task_id not in scheduler._local_status_index[TaskStatus.PENDING]
        assert task_id in scheduler._local_status_index[TaskStatus.RUNNING]


class TestRegionalClaimExtended:
    """Extended tests for regional claim functionality."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler."""
        return TaskScheduler(redis_url="memory://")

    @pytest.mark.asyncio
    async def test_claim_in_region_fallback_region(self, scheduler):
        """Test claiming task in fallback region."""
        await scheduler.submit(
            task_type="regional",
            payload={},
            target_region="us-west-2",
            fallback_regions=["us-east-1", "eu-west-1"],
            region_routing_mode=RegionRoutingMode.PREFERRED,
        )

        # Worker in fallback region should get the task
        task = await scheduler.claim_in_region(
            worker_id="worker-east",
            capabilities=[],
            worker_region="us-east-1",
        )

        assert task is not None
        assert task.assigned_region == "us-east-1"

    @pytest.mark.asyncio
    async def test_claim_in_region_any_mode_accepts_all(self, scheduler):
        """Test ANY mode accepts any region."""
        await scheduler.submit(
            task_type="flexible",
            payload={},
            target_region="us-west-2",
            region_routing_mode=RegionRoutingMode.ANY,
        )

        # Worker in completely different region should get task
        task = await scheduler.claim_in_region(
            worker_id="worker-asia",
            capabilities=[],
            worker_region="ap-southeast-1",
        )

        assert task is not None
        assert task.assigned_region == "ap-southeast-1"

    @pytest.mark.asyncio
    async def test_claim_in_region_releases_on_mismatch(self, scheduler):
        """Test task is released back when region doesn't match."""
        task_id = await scheduler.submit(
            task_type="strict-region",
            payload={},
            target_region="us-west-2",
            region_routing_mode=RegionRoutingMode.STRICT,
        )

        # Worker in wrong region should not get task
        task = await scheduler.claim_in_region(
            worker_id="worker-wrong",
            capabilities=[],
            worker_region="eu-west-1",
        )

        assert task is None

        # Task should still be pending
        stored_task = await scheduler.get(task_id)
        assert stored_task.status == TaskStatus.PENDING


class TestTaskAcknowledgment:
    """Tests for task acknowledgment in Redis streams."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return create_mock_redis()

    @pytest.mark.asyncio
    async def test_ack_task_with_message_id(self, mock_redis):
        """Test task acknowledgment with stream message ID."""
        scheduler = TaskScheduler(redis_url="redis://localhost:6379")
        scheduler._redis = mock_redis

        task = Task(
            id="task-to-ack",
            task_type="test",
            payload={},
            priority=TaskPriority.NORMAL,
        )
        task.metadata["_stream_message_id"] = "1234567890-0"

        await scheduler._ack_task(task)

        mock_redis.xack.assert_called_once()
        mock_redis.xdel.assert_called_once()

    @pytest.mark.asyncio
    async def test_ack_task_without_message_id(self, mock_redis):
        """Test task acknowledgment without message ID logs warning."""
        scheduler = TaskScheduler(redis_url="redis://localhost:6379")
        scheduler._redis = mock_redis

        task = Task(
            id="task-no-msg",
            task_type="test",
            payload={},
        )
        # No _stream_message_id in metadata

        await scheduler._ack_task(task)

        # Should not call xack/xdel
        mock_redis.xack.assert_not_called()
        mock_redis.xdel.assert_not_called()

    @pytest.mark.asyncio
    async def test_ack_task_handles_connection_error(self, mock_redis):
        """Test acknowledgment handles connection errors gracefully."""
        mock_redis.xack.side_effect = ConnectionError("Redis unavailable")

        scheduler = TaskScheduler(redis_url="redis://localhost:6379")
        scheduler._redis = mock_redis

        task = Task(
            id="task-error",
            task_type="test",
            payload={},
        )
        task.metadata["_stream_message_id"] = "1234567890-0"

        # Should not raise
        await scheduler._ack_task(task)


class TestReleaseTask:
    """Tests for task release functionality."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler."""
        return TaskScheduler(redis_url="memory://")

    @pytest.mark.asyncio
    async def test_release_task_resets_status(self, scheduler):
        """Test releasing task resets status to PENDING."""
        task_id = await scheduler.submit(task_type="test", payload={})
        task = await scheduler.claim(worker_id="w1", capabilities=[])

        assert task.status == TaskStatus.RUNNING
        assert task.assigned_agent == "w1"

        await scheduler._release_task(task)

        # Reload task
        released_task = await scheduler.get(task_id)
        assert released_task.status == TaskStatus.PENDING
        assert released_task.assigned_agent is None
        assert released_task.assigned_at is None

    @pytest.mark.asyncio
    async def test_release_task_reenqueues(self, scheduler):
        """Test released task can be claimed again."""
        task_id = await scheduler.submit(task_type="test", payload={})
        task = await scheduler.claim(worker_id="w1", capabilities=[])

        await scheduler._release_task(task)

        # Another worker should be able to claim it
        task2 = await scheduler.claim(worker_id="w2", capabilities=[])
        assert task2 is not None
        assert task2.id == task_id
        assert task2.assigned_agent == "w2"


class TestCircuitBreaker:
    """Tests for circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_connect_respects_circuit_breaker(self):
        """Test connection respects circuit breaker state."""
        with patch("aragora.control_plane.scheduler._scheduler_redis_cb") as mock_cb:
            mock_cb.can_execute.return_value = False

            scheduler = TaskScheduler(redis_url="redis://localhost:6379")
            await scheduler.connect()

            # Should fall back to in-memory due to open circuit breaker
            assert scheduler._redis is None

    @pytest.mark.asyncio
    async def test_connect_records_success(self):
        """Test successful connection records success."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.xgroup_create = AsyncMock()

        with (
            patch("aragora.control_plane.scheduler._scheduler_redis_cb") as mock_cb,
            patch("redis.asyncio.from_url", return_value=mock_redis),
        ):
            mock_cb.can_execute.return_value = True

            scheduler = TaskScheduler(redis_url="redis://localhost:6379")
            await scheduler.connect()

            mock_cb.record_success.assert_called_once()


class TestWorkspaceIdHandling:
    """Tests for workspace ID in tasks."""

    @pytest.fixture
    def scheduler(self):
        """Create scheduler."""
        return TaskScheduler(redis_url="memory://")

    @pytest.mark.asyncio
    async def test_submit_stores_workspace_id(self, scheduler):
        """Test workspace ID is stored in task metadata."""
        task_id = await scheduler.submit(
            task_type="test",
            payload={},
            workspace_id="ws-123",
        )

        task = await scheduler.get(task_id)
        assert task.metadata["workspace_id"] == "ws-123"

    @pytest.mark.asyncio
    async def test_claim_uses_task_workspace_for_policy(self, scheduler):
        """Test claim uses task's workspace for policy check if not provided."""
        mock_policy_manager = MagicMock()
        mock_result = MagicMock()
        mock_result.allowed = True
        mock_policy_manager.evaluate_task_dispatch.return_value = mock_result

        scheduler._policy_manager = mock_policy_manager

        with patch("aragora.control_plane.scheduler.HAS_POLICY", True):
            await scheduler.submit(
                task_type="test",
                payload={},
                workspace_id="task-ws",
            )

            await scheduler.claim(worker_id="w1", capabilities=[])

            # Check policy was evaluated with task's workspace
            call_kwargs = mock_policy_manager.evaluate_task_dispatch.call_args[1]
            assert call_kwargs["workspace"] == "task-ws"
