"""
Workflow Task Queue with Dependency Resolution.

Provides asynchronous task scheduling for workflow execution with:
- Priority-based task ordering
- Dependency tracking and resolution
- Concurrent execution management
- Task lifecycle events

Usage:
    from aragora.workflow.queue import TaskQueue, WorkflowTask

    queue = TaskQueue(max_concurrent=5)
    await queue.start()

    # Add tasks with dependencies
    task1 = WorkflowTask(id="t1", workflow_id="wf1", step_id="step1")
    task2 = WorkflowTask(id="t2", workflow_id="wf1", step_id="step2", depends_on=["t1"])

    await queue.enqueue(task1)
    await queue.enqueue(task2)

    # Wait for completion
    await queue.wait_for_workflow("wf1")
"""

from aragora.workflow.queue.task import (
    TaskStatus,
    TaskPriority,
    WorkflowTask,
)
from aragora.workflow.queue.queue import (
    TaskQueue,
    TaskQueueConfig,
)
from aragora.workflow.queue.executor import (
    TaskExecutor,
    ExecutorPool,
)
from aragora.workflow.queue.scheduler import (
    DependencyScheduler,
    SchedulingPolicy,
)

__all__ = [
    # Task
    "TaskStatus",
    "TaskPriority",
    "WorkflowTask",
    # Queue
    "TaskQueue",
    "TaskQueueConfig",
    # Executor
    "TaskExecutor",
    "ExecutorPool",
    # Scheduler
    "DependencyScheduler",
    "SchedulingPolicy",
]
