"""Tests for DependencyScheduler and related classes."""

import asyncio
import pytest

from aragora.workflow.queue import (
    TaskQueue,
    TaskQueueConfig,
    WorkflowTask,
    TaskPriority,
    TaskStatus,
)
from aragora.workflow.queue.executor import ExecutorPool, PoolConfig
from aragora.workflow.queue.scheduler import (
    DependencyGraph,
    DependencyScheduler,
    SchedulerConfig,
    SchedulingPolicy,
    WorkflowState,
)


class TestDependencyGraph:
    """Tests for DependencyGraph."""

    @pytest.fixture
    def graph(self):
        """Create a dependency graph."""
        return DependencyGraph()

    def test_add_task_no_deps(self, graph):
        """Test adding a task without dependencies."""
        graph.add_task("task_1")
        assert "task_1" in graph._tasks

    def test_add_task_with_deps(self, graph):
        """Test adding a task with dependencies."""
        graph.add_task("task_1")
        graph.add_task("task_2", depends_on=["task_1"])

        assert "task_1" in graph._tasks
        assert "task_2" in graph._tasks
        assert "task_1" in graph._dependencies["task_2"]
        assert "task_2" in graph._dependents["task_1"]

    def test_remove_task(self, graph):
        """Test removing a task."""
        graph.add_task("task_1")
        graph.add_task("task_2", depends_on=["task_1"])

        graph.remove_task("task_2")

        assert "task_2" not in graph._tasks
        assert "task_2" not in graph._dependents["task_1"]

    def test_get_ready_tasks(self, graph):
        """Test getting ready tasks."""
        graph.add_task("task_1")
        graph.add_task("task_2")
        graph.add_task("task_3", depends_on=["task_1"])

        # Initially, task_1 and task_2 are ready (no deps)
        ready = graph.get_ready_tasks(completed=set())
        assert "task_1" in ready
        assert "task_2" in ready
        assert "task_3" not in ready

        # After task_1 completes, task_3 becomes ready
        ready = graph.get_ready_tasks(completed={"task_1"})
        assert "task_3" in ready
        assert "task_1" not in ready  # Already completed

    def test_get_dependencies(self, graph):
        """Test getting dependencies."""
        graph.add_task("task_1")
        graph.add_task("task_2", depends_on=["task_1"])

        deps = graph.get_dependencies("task_2")
        assert "task_1" in deps

    def test_get_dependents(self, graph):
        """Test getting dependents."""
        graph.add_task("task_1")
        graph.add_task("task_2", depends_on=["task_1"])
        graph.add_task("task_3", depends_on=["task_1"])

        dependents = graph.get_dependents("task_1")
        assert "task_2" in dependents
        assert "task_3" in dependents

    def test_has_cycle_no_cycle(self, graph):
        """Test cycle detection with no cycle."""
        graph.add_task("task_1")
        graph.add_task("task_2", depends_on=["task_1"])
        graph.add_task("task_3", depends_on=["task_2"])

        assert not graph.has_cycle()

    def test_has_cycle_with_cycle(self, graph):
        """Test cycle detection with a cycle."""
        graph.add_task("task_1", depends_on=["task_3"])
        graph.add_task("task_2", depends_on=["task_1"])
        graph.add_task("task_3", depends_on=["task_2"])

        assert graph.has_cycle()

    def test_topological_sort(self, graph):
        """Test topological sorting."""
        graph.add_task("task_1")
        graph.add_task("task_2", depends_on=["task_1"])
        graph.add_task("task_3", depends_on=["task_1"])
        graph.add_task("task_4", depends_on=["task_2", "task_3"])

        order = graph.topological_sort()

        # task_1 must come before task_2, task_3
        # task_2, task_3 must come before task_4
        assert order.index("task_1") < order.index("task_2")
        assert order.index("task_1") < order.index("task_3")
        assert order.index("task_2") < order.index("task_4")
        assert order.index("task_3") < order.index("task_4")


class TestWorkflowState:
    """Tests for WorkflowState."""

    def test_workflow_state(self):
        """Test workflow state tracking."""
        state = WorkflowState(
            workflow_id="wf_001",
            task_ids={"t1", "t2", "t3"},
            pending_count=2,
            running_count=1,
            completed_count=0,
        )

        assert not state.is_complete
        assert state.progress == 0.0

    def test_workflow_complete(self):
        """Test workflow completion detection."""
        state = WorkflowState(
            workflow_id="wf_001",
            task_ids={"t1", "t2"},
            pending_count=0,
            running_count=0,
            completed_count=2,
        )

        assert state.is_complete
        assert state.progress == 1.0

    def test_workflow_progress(self):
        """Test workflow progress calculation."""
        state = WorkflowState(
            workflow_id="wf_001",
            task_ids={"t1", "t2", "t3", "t4"},
            pending_count=2,
            running_count=1,
            completed_count=1,
        )

        assert state.progress == 0.25  # 1 of 4 completed


class TestSchedulingPolicy:
    """Tests for SchedulingPolicy enum."""

    def test_policy_values(self):
        """Test scheduling policy values."""
        assert SchedulingPolicy.FIFO.value == "fifo"
        assert SchedulingPolicy.PRIORITY.value == "priority"
        assert SchedulingPolicy.SHORTEST_FIRST.value == "shortest_first"
        assert SchedulingPolicy.DEADLINE.value == "deadline"
        assert SchedulingPolicy.FAIR.value == "fair"


class TestDependencyScheduler:
    """Tests for DependencyScheduler."""

    @pytest.fixture
    def scheduler_config(self):
        """Create scheduler configuration."""
        return SchedulerConfig(
            policy=SchedulingPolicy.PRIORITY,
            max_concurrent_workflows=5,
            max_tasks_per_workflow=10,
        )

    @pytest.fixture
    def scheduler(self, scheduler_config):
        """Create a scheduler."""
        queue = TaskQueue(config=TaskQueueConfig(max_concurrent=2))
        pool = ExecutorPool(config=PoolConfig(min_executors=2, max_executors=4))
        return DependencyScheduler(
            config=scheduler_config,
            queue=queue,
            executor_pool=pool,
        )

    @pytest.mark.asyncio
    async def test_start_stop(self, scheduler):
        """Test starting and stopping the scheduler."""
        await scheduler.start()
        assert scheduler._started

        await scheduler.stop()
        assert not scheduler._started

    @pytest.mark.asyncio
    async def test_submit_workflow(self, scheduler):
        """Test submitting a workflow."""
        await scheduler.start()
        try:
            tasks = [
                WorkflowTask.create(workflow_id="wf_001", step_id="step_1"),
                WorkflowTask.create(workflow_id="wf_001", step_id="step_2"),
            ]

            workflow_id = await scheduler.submit_workflow("wf_001", tasks)

            assert workflow_id == "wf_001"
            assert "wf_001" in scheduler._workflows
            assert len(scheduler._workflows["wf_001"].task_ids) == 2
        finally:
            await scheduler.stop(drain=False)

    @pytest.mark.asyncio
    async def test_submit_workflow_with_dependencies(self, scheduler):
        """Test submitting a workflow with task dependencies."""
        await scheduler.start()
        try:
            task1 = WorkflowTask.create(workflow_id="wf_001", step_id="step_1")
            task2 = WorkflowTask.create(
                workflow_id="wf_001",
                step_id="step_2",
                depends_on=[task1.id],
            )

            await scheduler.submit_workflow("wf_001", [task1, task2])

            # Only task1 should be initially scheduled (no dependencies)
            state = scheduler.get_workflow_state("wf_001")
            assert state is not None
        finally:
            await scheduler.stop(drain=False)

    @pytest.mark.asyncio
    async def test_reject_cyclic_dependencies(self, scheduler):
        """Test that cyclic dependencies are rejected."""
        await scheduler.start()
        try:
            task1 = WorkflowTask.create(
                workflow_id="wf_001",
                step_id="step_1",
                depends_on=["task_fake"],  # Will create a cycle with task2
            )
            task2 = WorkflowTask.create(
                workflow_id="wf_001",
                step_id="step_2",
                depends_on=[task1.id],
            )

            # Manually create a cycle
            task1.depends_on = [task2.id]

            with pytest.raises(ValueError, match="circular dependencies"):
                await scheduler.submit_workflow("wf_001", [task1, task2])
        finally:
            await scheduler.stop(drain=False)

    @pytest.mark.asyncio
    async def test_max_concurrent_workflows(self, scheduler):
        """Test maximum concurrent workflows limit."""
        scheduler._config.max_concurrent_workflows = 2

        await scheduler.start()
        try:
            # Submit max workflows
            for i in range(2):
                tasks = [WorkflowTask.create(workflow_id=f"wf_{i}", step_id="step_1")]
                await scheduler.submit_workflow(f"wf_{i}", tasks)

            # Third should fail
            with pytest.raises(RuntimeError, match="Maximum concurrent workflows"):
                tasks = [WorkflowTask.create(workflow_id="wf_2", step_id="step_1")]
                await scheduler.submit_workflow("wf_2", tasks)
        finally:
            await scheduler.stop(drain=False)

    @pytest.mark.asyncio
    async def test_get_workflow_state(self, scheduler):
        """Test getting workflow state."""
        await scheduler.start()
        try:
            tasks = [
                WorkflowTask.create(workflow_id="wf_001", step_id="step_1"),
            ]
            await scheduler.submit_workflow("wf_001", tasks)

            state = scheduler.get_workflow_state("wf_001")
            assert state is not None
            assert state.workflow_id == "wf_001"

            # Non-existent workflow
            assert scheduler.get_workflow_state("nonexistent") is None
        finally:
            await scheduler.stop(drain=False)

    @pytest.mark.asyncio
    async def test_get_workflow_progress(self, scheduler):
        """Test getting workflow progress."""
        await scheduler.start()
        try:
            tasks = [
                WorkflowTask.create(workflow_id="wf_001", step_id="step_1"),
            ]
            await scheduler.submit_workflow("wf_001", tasks)

            progress = scheduler.get_workflow_progress("wf_001")
            assert 0.0 <= progress <= 1.0

            # Non-existent workflow
            assert scheduler.get_workflow_progress("nonexistent") == 0.0
        finally:
            await scheduler.stop(drain=False)

    @pytest.mark.asyncio
    async def test_cancel_workflow(self, scheduler):
        """Test cancelling a workflow."""
        await scheduler.start()
        try:
            tasks = [
                WorkflowTask.create(workflow_id="wf_001", step_id="step_1"),
                WorkflowTask.create(workflow_id="wf_001", step_id="step_2"),
            ]
            await scheduler.submit_workflow("wf_001", tasks)

            cancelled = await scheduler.cancel_workflow("wf_001")
            assert cancelled >= 0
        finally:
            await scheduler.stop(drain=False)

    @pytest.mark.asyncio
    async def test_get_stats(self, scheduler):
        """Test getting scheduler statistics."""
        await scheduler.start()
        try:
            stats = scheduler.get_stats()

            assert "policy" in stats
            assert stats["policy"] == "priority"
            assert "active_workflows" in stats
            assert "total_workflows" in stats
            assert "queue" in stats
            assert "executor_pool" in stats
        finally:
            await scheduler.stop(drain=False)

    @pytest.mark.asyncio
    async def test_workflow_completion_callback(self, scheduler):
        """Test workflow completion callback."""
        completed_workflows = []

        def on_complete(workflow_id: str):
            completed_workflows.append(workflow_id)

        scheduler.on_workflow_complete(on_complete)

        await scheduler.start()
        try:
            # The callback would be called when workflow completes
            # This is more of an integration test
            pass
        finally:
            await scheduler.stop(drain=False)
