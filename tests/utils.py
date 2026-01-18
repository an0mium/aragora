"""
Test utilities for Aragora test suite.

Provides helper functions and context managers for test setup and teardown.
"""

import logging
from contextlib import contextmanager
from typing import Any, Generator


logger = logging.getLogger(__name__)


@contextmanager
def managed_fixture(instance: Any, name: str = "fixture") -> Generator[Any, None, None]:
    """
    Context manager for fixtures that need cleanup.

    Ensures proper resource cleanup even if tests fail.

    Args:
        instance: The fixture instance to manage
        name: Name for logging purposes

    Yields:
        The instance

    Example:
        with managed_fixture(EloSystem(db_path=temp_db), "EloSystem") as elo:
            yield elo
    """
    try:
        yield instance
    finally:
        # Try common cleanup methods
        cleanup_methods = ["close", "cleanup", "shutdown", "dispose", "__del__"]
        for method in cleanup_methods:
            if hasattr(instance, method):
                try:
                    cleanup = getattr(instance, method)
                    if callable(cleanup):
                        cleanup()
                        logger.debug(f"{name} cleaned up via {method}()")
                        break
                except Exception as e:
                    logger.warning(f"Error cleaning up {name} via {method}(): {e}")


def create_mock_agent_info(
    agent_id: str = "test-agent",
    capabilities: list[str] | None = None,
    model: str = "test-model",
    provider: str = "test-provider",
    status: str = "ready",
) -> dict:
    """
    Create a mock agent info dictionary for testing.

    Args:
        agent_id: Agent identifier
        capabilities: List of capabilities
        model: Model name
        provider: Provider name
        status: Agent status

    Returns:
        Dictionary matching AgentInfo structure
    """
    import time

    return {
        "agent_id": agent_id,
        "capabilities": capabilities or ["debate", "code"],
        "status": status,
        "model": model,
        "provider": provider,
        "metadata": {},
        "registered_at": time.time(),
        "last_heartbeat": time.time(),
        "current_task_id": None,
        "tasks_completed": 0,
        "tasks_failed": 0,
        "avg_latency_ms": 0.0,
        "tags": [],
    }


def create_mock_task(
    task_type: str = "debate",
    payload: dict | None = None,
    required_capabilities: list[str] | None = None,
    priority: int = 50,
    status: str = "pending",
) -> dict:
    """
    Create a mock task dictionary for testing.

    Args:
        task_type: Type of task
        payload: Task payload
        required_capabilities: Required capabilities
        priority: Task priority (0-100)
        status: Task status

    Returns:
        Dictionary matching Task structure
    """
    import time
    import uuid

    return {
        "id": str(uuid.uuid4()),
        "task_type": task_type,
        "payload": payload or {"question": "Test question"},
        "required_capabilities": required_capabilities or [],
        "status": status,
        "priority": priority,
        "created_at": time.time(),
        "assigned_at": None,
        "started_at": None,
        "completed_at": None,
        "assigned_agent": None,
        "timeout_seconds": 300.0,
        "max_retries": 3,
        "retries": 0,
        "result": None,
        "error": None,
        "metadata": {},
    }
