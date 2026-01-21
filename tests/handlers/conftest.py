"""Fixtures for handler tests."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


class AgentStatus(Enum):
    """Mock agent status enum."""

    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


class TaskStatus(Enum):
    """Mock task status enum."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Mock task priority enum."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """Mock health status enum."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class MockAgentInfo:
    """Mock agent info."""

    agent_id: str
    capabilities: List[str]
    model: str
    provider: str
    status: AgentStatus
    last_heartbeat: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "capabilities": self.capabilities,
            "model": self.model,
            "provider": self.provider,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class MockTask:
    """Mock task."""

    task_id: str
    task_type: str
    payload: Dict[str, Any]
    status: TaskStatus
    priority: TaskPriority
    required_capabilities: List[str]
    assigned_agent: Optional[str]
    result: Optional[Any]
    error: Optional[str]
    created_at: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "payload": self.payload,
            "status": self.status.value,
            "priority": self.priority.value,
            "required_capabilities": self.required_capabilities,
            "assigned_agent": self.assigned_agent,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class MockHealthCheck:
    """Mock health check."""

    agent_id: str
    status: HealthStatus
    last_check: datetime
    latency_ms: float
    error_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "latency_ms": self.latency_ms,
            "error_rate": self.error_rate,
        }


class MockCoordinator:
    """Mock control plane coordinator for testing."""

    def __init__(self):
        self._agents: Dict[str, MockAgentInfo] = {}
        self._tasks: Dict[str, MockTask] = {}
        self._health: Dict[str, MockHealthCheck] = {}
        self._health_monitor = MagicMock()
        self._health_monitor.get_all_health.return_value = {}

    async def register_agent(
        self,
        agent_id: str,
        capabilities: List[str],
        model: str,
        provider: str,
        metadata: Dict[str, Any] = None,
    ) -> MockAgentInfo:
        """Register an agent."""
        agent = MockAgentInfo(
            agent_id=agent_id,
            capabilities=capabilities,
            model=model,
            provider=provider,
            status=AgentStatus.IDLE,
            last_heartbeat=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        self._agents[agent_id] = agent
        return agent

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    async def get_agent(self, agent_id: str) -> Optional[MockAgentInfo]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    async def list_agents(
        self,
        capability: Optional[str] = None,
        only_available: bool = True,
    ) -> List[MockAgentInfo]:
        """List agents."""
        agents = list(self._agents.values())
        if capability:
            agents = [a for a in agents if capability in a.capabilities]
        if only_available:
            agents = [a for a in agents if a.status != AgentStatus.OFFLINE]
        return agents

    async def heartbeat(self, agent_id: str, status: Optional[AgentStatus] = None) -> bool:
        """Process agent heartbeat."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False
        agent.last_heartbeat = datetime.now(timezone.utc)
        if status:
            agent.status = status
        return True

    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        required_capabilities: List[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: Optional[int] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Submit a task."""
        import uuid

        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task = MockTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            status=TaskStatus.PENDING,
            priority=priority,
            required_capabilities=required_capabilities or [],
            assigned_agent=None,
            result=None,
            error=None,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        self._tasks[task_id] = task
        return task_id

    async def get_task(self, task_id: str) -> Optional[MockTask]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    async def claim_task(
        self,
        agent_id: str,
        capabilities: List[str],
        block_ms: int = 5000,
    ) -> Optional[MockTask]:
        """Claim a task for an agent."""
        for task in self._tasks.values():
            if task.status == TaskStatus.PENDING:
                # Check capabilities match
                if not task.required_capabilities or any(
                    c in capabilities for c in task.required_capabilities
                ):
                    task.status = TaskStatus.RUNNING
                    task.assigned_agent = agent_id
                    return task
        return None

    async def complete_task(
        self,
        task_id: str,
        result: Any = None,
        agent_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> bool:
        """Complete a task."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        task.status = TaskStatus.COMPLETED
        task.result = result
        return True

    async def fail_task(
        self,
        task_id: str,
        error: str,
        agent_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        requeue: bool = True,
    ) -> bool:
        """Fail a task."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        if requeue:
            task.status = TaskStatus.PENDING
            task.assigned_agent = None
        else:
            task.status = TaskStatus.FAILED
        task.error = error
        return True

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self._tasks.get(task_id)
        if not task or task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            return False
        task.status = TaskStatus.CANCELLED
        return True

    def get_system_health(self) -> HealthStatus:
        """Get overall system health."""
        if not self._agents:
            return HealthStatus.DEGRADED
        healthy_count = sum(1 for h in self._health.values() if h.status == HealthStatus.HEALTHY)
        if healthy_count == 0:
            return HealthStatus.UNHEALTHY
        if healthy_count < len(self._agents):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def get_agent_health(self, agent_id: str) -> Optional[MockHealthCheck]:
        """Get agent health."""
        return self._health.get(agent_id)

    async def get_stats(self) -> Dict[str, Any]:
        """Get control plane statistics."""
        return {
            "agents": {
                "total": len(self._agents),
                "idle": sum(1 for a in self._agents.values() if a.status == AgentStatus.IDLE),
                "busy": sum(1 for a in self._agents.values() if a.status == AgentStatus.BUSY),
            },
            "tasks": {
                "total": len(self._tasks),
                "pending": sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING),
                "running": sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING),
                "completed": sum(
                    1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED
                ),
            },
        }


class MockRequest:
    """Mock HTTP request."""

    def __init__(
        self,
        method: str = "GET",
        path: str = "/",
        query: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
    ):
        self.method = method
        self.path = path
        self.query = query or {}
        self._body = body

    async def json(self) -> Dict[str, Any]:
        """Get JSON body."""
        return self._body or {}

    async def body(self) -> bytes:
        """Get raw body."""
        import json

        return json.dumps(self._body or {}).encode()


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, body: Optional[Dict[str, Any]] = None):
        self.rfile = MagicMock()
        self._body = body
        if body:
            import json

            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {"Content-Length": str(len(body_bytes))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator."""
    return MockCoordinator()


@pytest.fixture
def mock_request():
    """Factory for creating mock requests."""

    def _create_request(
        method: str = "GET",
        path: str = "/",
        query: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> MockRequest:
        return MockRequest(method=method, path=path, query=query, body=body)

    return _create_request


@pytest.fixture
def mock_handler():
    """Factory for creating mock handlers."""

    def _create_handler(body: Optional[Dict[str, Any]] = None) -> MockHandler:
        return MockHandler(body=body)

    return _create_handler
