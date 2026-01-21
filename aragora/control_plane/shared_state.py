"""
Shared Control Plane State for unified backing across handlers.

Provides a singleton-style shared state that bridges:
- AgentDashboardHandler (UI) with persistent control plane components
- Redis-backed or in-memory persistence depending on availability

This enables multiple server instances to share control plane state
when Redis is available, while gracefully falling back to in-memory
for single-instance development deployments.

Usage:
    from aragora.control_plane.shared_state import (
        get_shared_state,
        set_shared_state,
        SharedControlPlaneState,
    )

    # Initialize with Redis (production)
    state = SharedControlPlaneState(redis_url="redis://localhost:6379")
    await state.connect()
    set_shared_state(state)

    # Or use defaults (auto-detects Redis or falls back to in-memory)
    state = await get_shared_state()

    # Use in handlers
    agents = await state.list_agents()
    await state.update_agent_status(agent_id, "paused")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Module-level shared state singleton
_shared_state: Optional["SharedControlPlaneState"] = None


@dataclass
class AgentState:
    """
    Unified agent state for UI and control plane.

    Compatible with both AgentDashboardHandler's dict format and
    AgentRegistry's AgentInfo.
    """

    id: str
    name: str
    type: str
    model: str
    status: str  # "active", "paused", "idle", "offline"
    role: str = ""
    capabilities: Set[str] = field(default_factory=set)
    tasks_completed: int = 0
    findings_generated: int = 0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0
    created_at: str = ""
    last_active: Optional[str] = None
    paused_at: Optional[str] = None
    resumed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "model": self.model,
            "status": self.status,
            "role": self.role,
            "capabilities": list(self.capabilities),
            "tasks_completed": self.tasks_completed,
            "findings_generated": self.findings_generated,
            "avg_response_time": self.avg_response_time,
            "error_rate": self.error_rate,
            "uptime_seconds": self.uptime_seconds,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "paused_at": self.paused_at,
            "resumed_at": self.resumed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Create from dict."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            type=data.get("type", ""),
            model=data.get("model", ""),
            status=data.get("status", "idle"),
            role=data.get("role", ""),
            capabilities=set(data.get("capabilities", [])),
            tasks_completed=data.get("tasks_completed", 0),
            findings_generated=data.get("findings_generated", 0),
            avg_response_time=data.get("avg_response_time", 0.0),
            error_rate=data.get("error_rate", 0.0),
            uptime_seconds=data.get("uptime_seconds", 0.0),
            created_at=data.get("created_at", ""),
            last_active=data.get("last_active"),
            paused_at=data.get("paused_at"),
            resumed_at=data.get("resumed_at"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TaskState:
    """
    Unified task state for UI and control plane.

    Compatible with both AgentDashboardHandler's dict format and
    TaskScheduler's Task.
    """

    id: str
    type: str
    priority: str  # "high", "normal", "low"
    status: str  # "pending", "processing", "completed", "failed"
    created_at: str = ""
    assigned_agent: Optional[str] = None
    document_id: Optional[str] = None
    audit_type: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "assigned_agent": self.assigned_agent,
            "document_id": self.document_id,
            "audit_type": self.audit_type,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskState":
        """Create from dict."""
        return cls(
            id=data.get("id", ""),
            type=data.get("type", ""),
            priority=data.get("priority", "normal"),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", ""),
            assigned_agent=data.get("assigned_agent"),
            document_id=data.get("document_id"),
            audit_type=data.get("audit_type"),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
        )


class SharedControlPlaneState:
    """
    Shared state for control plane with Redis persistence.

    Provides unified interface for:
    - Agent state management (list, pause, resume, update)
    - Task queue management (list, prioritize, claim)
    - Metrics aggregation

    When Redis is available, state is shared across server instances.
    Falls back to SQLite for durable single-instance deployments (no data loss on restart).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "aragora:cp:shared:",
        heartbeat_timeout: float = 30.0,
        sqlite_path: Optional[str] = None,
    ):
        """
        Initialize shared state.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for Redis keys
            heartbeat_timeout: Seconds before agent is considered offline
            sqlite_path: Optional SQLite database path for fallback storage
        """
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._heartbeat_timeout = heartbeat_timeout
        self._redis: Optional[Any] = None
        self._connected = False

        # SQLite fallback path
        if sqlite_path is None:
            data_dir = os.environ.get("ARAGORA_DATA_DIR", ".nomic")
            sqlite_path = str(Path(data_dir) / "control_plane_state.db")
        self._sqlite_path = sqlite_path
        self._sqlite_initialized = False

        # In-memory cache (backed by SQLite when Redis unavailable)
        self._local_agents: Dict[str, AgentState] = {}
        self._local_tasks: List[TaskState] = []
        self._local_metrics: Dict[str, Any] = {
            "total_tasks_processed": 0,
            "total_findings_generated": 0,
            "active_sessions": 0,
            "agent_uptime": {},
        }
        self._stream_clients: List[asyncio.Queue] = []

    @property
    def is_persistent(self) -> bool:
        """Check if using persistent (Redis or SQLite) backing."""
        return (self._redis is not None and self._connected) or self._sqlite_initialized

    @property
    def is_redis_connected(self) -> bool:
        """Check if connected to Redis (multi-instance mode)."""
        return self._redis is not None and self._connected

    def _init_sqlite(self) -> None:
        """Initialize SQLite database for fallback persistence."""
        try:
            Path(self._sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._sqlite_path)

            # Agents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS control_plane_agents (
                    id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

            # Tasks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS control_plane_tasks (
                    id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    position INTEGER,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cp_tasks_priority ON control_plane_tasks(priority, position)"
            )

            # Metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS control_plane_metrics (
                    key TEXT PRIMARY KEY,
                    value_int INTEGER DEFAULT 0,
                    updated_at REAL NOT NULL
                )
            """)

            conn.commit()
            conn.close()
            self._sqlite_initialized = True
            logger.info(f"SharedControlPlaneState SQLite initialized: {self._sqlite_path}")

            # Load existing state into memory cache
            self._load_sqlite_state()

        except Exception as e:
            logger.warning(f"Failed to initialize SQLite fallback: {e}")
            self._sqlite_initialized = False

    def _load_sqlite_state(self) -> None:
        """Load existing state from SQLite into memory cache."""
        if not self._sqlite_initialized:
            return

        try:
            conn = sqlite3.connect(self._sqlite_path)

            # Load agents
            cursor = conn.execute("SELECT id, data_json FROM control_plane_agents")
            for row in cursor.fetchall():
                try:
                    agent = AgentState.from_dict(json.loads(row[1]))
                    self._local_agents[row[0]] = agent
                except (json.JSONDecodeError, KeyError):
                    pass

            # Load tasks
            cursor = conn.execute(
                "SELECT id, data_json FROM control_plane_tasks ORDER BY priority, position"
            )
            self._local_tasks = []
            for row in cursor.fetchall():
                try:
                    task = TaskState.from_dict(json.loads(row[1]))
                    self._local_tasks.append(task)
                except (json.JSONDecodeError, KeyError):
                    pass

            # Load metrics
            cursor = conn.execute("SELECT key, value_int FROM control_plane_metrics")
            for row in cursor.fetchall():
                if row[0] in self._local_metrics:
                    self._local_metrics[row[0]] = row[1]

            conn.close()
            logger.debug(
                f"Loaded {len(self._local_agents)} agents, {len(self._local_tasks)} tasks from SQLite"
            )

        except Exception as e:
            logger.warning(f"Failed to load state from SQLite: {e}")

    async def connect(self) -> bool:
        """
        Connect to Redis backend. Falls back to SQLite if Redis unavailable.

        Returns:
            True if connected to Redis, False if using SQLite fallback
        """
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()
            self._connected = True
            logger.info(f"SharedControlPlaneState connected to Redis: {self._redis_url}")
            return True

        except ImportError:
            logger.warning("redis package not installed, using SQLite fallback")
            self._redis = None
            self._connected = False
            self._init_sqlite()
            return False
        except Exception as e:
            logger.warning(f"Failed to connect to Redis, using SQLite fallback: {e}")
            self._redis = None
            self._connected = False
            self._init_sqlite()
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._connected = False
            logger.info("SharedControlPlaneState disconnected from Redis")

    # --- Agent Management ---

    async def list_agents(
        self,
        status_filter: Optional[str] = None,
        type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all agents with optional filtering.

        Args:
            status_filter: Filter by status ("active", "paused", "idle")
            type_filter: Filter by agent type

        Returns:
            List of agent dicts
        """
        agents = await self._get_all_agents()

        # Apply filters
        if status_filter:
            agents = [a for a in agents if a.status == status_filter]
        if type_filter:
            agents = [a for a in agents if a.type == type_filter]

        return [a.to_dict() for a in agents]

    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent by ID."""
        agent = await self._get_agent(agent_id)
        return agent.to_dict() if agent else None

    async def register_agent(self, agent_data: Dict[str, Any]) -> AgentState:
        """Register or update an agent."""
        agent = AgentState.from_dict(agent_data)
        if not agent.created_at:
            agent.created_at = datetime.utcnow().isoformat()
        agent.last_active = datetime.utcnow().isoformat()

        await self._save_agent(agent)
        await self._broadcast_event({
            "type": "agent_registered",
            "agent_id": agent.id,
            "timestamp": datetime.utcnow().isoformat(),
        })
        return agent

    async def update_agent_status(
        self,
        agent_id: str,
        status: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Update agent status (pause/resume/etc).

        Args:
            agent_id: Agent to update
            status: New status

        Returns:
            Updated agent dict or None if not found
        """
        agent = await self._get_agent(agent_id)
        if not agent:
            return None

        old_status = agent.status
        agent.status = status

        if status == "paused":
            agent.paused_at = datetime.utcnow().isoformat()
        elif status == "active" and old_status == "paused":
            agent.resumed_at = datetime.utcnow().isoformat()
            agent.paused_at = None

        agent.last_active = datetime.utcnow().isoformat()
        await self._save_agent(agent)

        await self._broadcast_event({
            "type": f"agent_{status}" if status in ("paused", "resumed") else "agent_status_changed",
            "agent_id": agent_id,
            "old_status": old_status,
            "new_status": status,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return agent.to_dict()

    async def record_agent_activity(
        self,
        agent_id: str,
        tasks_completed: int = 0,
        findings_generated: int = 0,
        response_time_ms: Optional[float] = None,
        error: bool = False,
    ) -> None:
        """Record activity metrics for an agent."""
        agent = await self._get_agent(agent_id)
        if not agent:
            return

        agent.tasks_completed += tasks_completed
        agent.findings_generated += findings_generated
        agent.last_active = datetime.utcnow().isoformat()

        if response_time_ms is not None:
            # Rolling average
            total = agent.tasks_completed
            if total > 0:
                agent.avg_response_time = (
                    (agent.avg_response_time * (total - 1) + response_time_ms) / total
                )

        if error:
            # Simple error rate calculation
            total_tasks = agent.tasks_completed + 1
            errors = int(agent.error_rate * (total_tasks - 1)) + 1
            agent.error_rate = errors / total_tasks

        await self._save_agent(agent)

    # --- Task Queue Management ---

    async def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks in the queue."""
        tasks = await self._get_all_tasks()
        return [t.to_dict() for t in tasks]

    async def add_task(self, task_data: Dict[str, Any]) -> TaskState:
        """Add a task to the queue."""
        task = TaskState.from_dict(task_data)
        if not task.created_at:
            task.created_at = datetime.utcnow().isoformat()

        await self._save_task(task)
        await self._broadcast_event({
            "type": "task_added",
            "task_id": task.id,
            "timestamp": datetime.utcnow().isoformat(),
        })
        return task

    async def update_task_priority(
        self,
        task_id: str,
        priority: str,
        position: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update task priority and/or position."""
        task = await self._get_task(task_id)
        if not task:
            return None

        task.priority = priority
        await self._save_task(task, position=position)

        await self._broadcast_event({
            "type": "queue_updated",
            "task_id": task_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return task.to_dict()

    # --- Metrics ---

    async def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics."""
        agents = await self._get_all_agents()
        tasks = await self._get_all_tasks()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "agents": {
                "total": len(agents),
                "active": sum(1 for a in agents if a.status == "active"),
                "paused": sum(1 for a in agents if a.status == "paused"),
                "idle": sum(1 for a in agents if a.status == "idle"),
            },
            "queue": {
                "total_tasks": len(tasks),
                "pending": sum(1 for t in tasks if t.status == "pending"),
                "processing": sum(1 for t in tasks if t.status == "processing"),
            },
            "processing": {
                "total_tasks_processed": sum(a.tasks_completed for a in agents),
                "total_findings_generated": sum(a.findings_generated for a in agents),
            },
            "performance": {
                "avg_task_duration_ms": self._calculate_avg_duration(agents),
                "error_rate": self._calculate_error_rate(agents),
            },
        }

    async def increment_metric(self, metric_name: str, value: int = 1) -> None:
        """Increment a global metric counter."""
        if self._redis:
            key = f"{self._key_prefix}metrics:{metric_name}"
            await self._redis.incrby(key, value)
        else:
            if metric_name in self._local_metrics:
                self._local_metrics[metric_name] += value

            # Persist to SQLite
            if self._sqlite_initialized:
                try:
                    conn = sqlite3.connect(self._sqlite_path)
                    conn.execute(
                        """INSERT INTO control_plane_metrics (key, value_int, updated_at)
                           VALUES (?, ?, ?)
                           ON CONFLICT(key) DO UPDATE SET
                           value_int = value_int + ?, updated_at = ?""",
                        (metric_name, value, time.time(), value, time.time()),
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.debug(f"Failed to persist metric to SQLite: {e}")

    # --- Stream/Events ---

    def register_stream_client(self, queue: asyncio.Queue) -> None:
        """Register a client for event streaming."""
        self._stream_clients.append(queue)

    def unregister_stream_client(self, queue: asyncio.Queue) -> None:
        """Unregister a stream client."""
        if queue in self._stream_clients:
            self._stream_clients.remove(queue)

    async def _broadcast_event(self, event: Dict[str, Any]) -> None:
        """Broadcast event to all stream clients."""
        # Broadcast locally
        for queue in self._stream_clients:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

        # Publish to Redis for cross-instance events
        if self._redis:
            try:
                channel = f"{self._key_prefix}events"
                await self._redis.publish(channel, json.dumps(event))
            except Exception as e:
                logger.warning(f"Failed to publish event to Redis: {e}")

    # --- Internal Storage Methods ---

    async def _get_agent(self, agent_id: str) -> Optional[AgentState]:
        """Get agent from storage."""
        if self._redis:
            key = f"{self._key_prefix}agents:{agent_id}"
            data = await self._redis.get(key)
            if data:
                return AgentState.from_dict(json.loads(data))
            return None
        else:
            return self._local_agents.get(agent_id)

    async def _get_all_agents(self) -> List[AgentState]:
        """Get all agents from storage."""
        if self._redis:
            agents = []
            pattern = f"{self._key_prefix}agents:*"
            async for key in self._redis.scan_iter(match=pattern):
                data = await self._redis.get(key)
                if data:
                    agents.append(AgentState.from_dict(json.loads(data)))
            return agents
        else:
            return list(self._local_agents.values())

    async def _save_agent(self, agent: AgentState) -> None:
        """Save agent to storage (Redis or SQLite)."""
        if self._redis:
            key = f"{self._key_prefix}agents:{agent.id}"
            await self._redis.set(
                key,
                json.dumps(agent.to_dict()),
                ex=86400,  # 24 hour expiry
            )
        else:
            # Save to memory cache
            self._local_agents[agent.id] = agent

            # Persist to SQLite
            if self._sqlite_initialized:
                try:
                    conn = sqlite3.connect(self._sqlite_path)
                    conn.execute(
                        """INSERT OR REPLACE INTO control_plane_agents
                           (id, data_json, updated_at) VALUES (?, ?, ?)""",
                        (agent.id, json.dumps(agent.to_dict()), time.time()),
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.debug(f"Failed to persist agent to SQLite: {e}")

    async def _get_task(self, task_id: str) -> Optional[TaskState]:
        """Get task from storage."""
        if self._redis:
            key = f"{self._key_prefix}tasks:{task_id}"
            data = await self._redis.get(key)
            if data:
                return TaskState.from_dict(json.loads(data))
            return None
        else:
            for task in self._local_tasks:
                if task.id == task_id:
                    return task
            return None

    async def _get_all_tasks(self) -> List[TaskState]:
        """Get all tasks from storage."""
        if self._redis:
            tasks = []
            pattern = f"{self._key_prefix}tasks:*"
            async for key in self._redis.scan_iter(match=pattern):
                data = await self._redis.get(key)
                if data:
                    tasks.append(TaskState.from_dict(json.loads(data)))
            # Sort by priority
            priority_order = {"high": 0, "normal": 1, "low": 2}
            tasks.sort(key=lambda t: priority_order.get(t.priority, 1))
            return tasks
        else:
            return list(self._local_tasks)

    async def _save_task(self, task: TaskState, position: Optional[int] = None) -> None:
        """Save task to storage (Redis or SQLite)."""
        if self._redis:
            key = f"{self._key_prefix}tasks:{task.id}"
            await self._redis.set(
                key,
                json.dumps(task.to_dict()),
                ex=86400,  # 24 hour expiry
            )
            # Also add to sorted set for ordering
            score = {"high": 0, "normal": 1, "low": 2}.get(task.priority, 1)
            if position is not None:
                score = position
            await self._redis.zadd(
                f"{self._key_prefix}task_order",
                {task.id: score},
            )
        else:
            # Update or add to local list
            found = False
            for i, t in enumerate(self._local_tasks):
                if t.id == task.id:
                    if position is not None and position != i:
                        self._local_tasks.pop(i)
                        self._local_tasks.insert(min(position, len(self._local_tasks)), task)
                    else:
                        self._local_tasks[i] = task
                    found = True
                    break
            if not found:
                if position is not None:
                    self._local_tasks.insert(min(position, len(self._local_tasks)), task)
                else:
                    self._local_tasks.append(task)
                # Sort by priority
                priority_order = {"high": 0, "normal": 1, "low": 2}
                self._local_tasks.sort(key=lambda t: priority_order.get(t.priority, 1))

            # Persist to SQLite
            if self._sqlite_initialized:
                try:
                    conn = sqlite3.connect(self._sqlite_path)
                    task_position = position if position is not None else self._local_tasks.index(task)
                    conn.execute(
                        """INSERT OR REPLACE INTO control_plane_tasks
                           (id, data_json, priority, position, updated_at) VALUES (?, ?, ?, ?, ?)""",
                        (task.id, json.dumps(task.to_dict()), task.priority, task_position, time.time()),
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.debug(f"Failed to persist task to SQLite: {e}")

    def _calculate_avg_duration(self, agents: List[AgentState]) -> float:
        """Calculate average task duration."""
        times = [a.avg_response_time for a in agents if a.avg_response_time > 0]
        return sum(times) / len(times) if times else 0.0

    def _calculate_error_rate(self, agents: List[AgentState]) -> float:
        """Calculate overall error rate."""
        rates = [a.error_rate for a in agents if a.error_rate > 0]
        return sum(rates) / len(rates) if rates else 0.0


# --- Module-level singleton functions ---


def set_shared_state(state: SharedControlPlaneState) -> None:
    """
    Set the global shared control plane state.

    Call this during application startup to configure the shared state
    with your preferred backing (Redis URL, etc.).

    Args:
        state: Configured SharedControlPlaneState instance
    """
    global _shared_state
    _shared_state = state
    logger.info(f"Set shared control plane state (persistent={state.is_persistent})")


def get_shared_state_sync() -> Optional[SharedControlPlaneState]:
    """
    Get the global shared state (synchronous version).

    Returns:
        SharedControlPlaneState or None if not initialized
    """
    return _shared_state


async def get_shared_state(
    redis_url: str = "redis://localhost:6379",
    auto_connect: bool = True,
) -> SharedControlPlaneState:
    """
    Get or create the global shared control plane state.

    If not already initialized, creates a new instance and optionally
    connects to Redis.

    Args:
        redis_url: Redis URL for new instance
        auto_connect: Whether to connect automatically

    Returns:
        SharedControlPlaneState instance
    """
    global _shared_state

    if _shared_state is None:
        _shared_state = SharedControlPlaneState(redis_url=redis_url)
        if auto_connect:
            await _shared_state.connect()
        logger.info(f"Created shared control plane state (persistent={_shared_state.is_persistent})")

    return _shared_state


async def close_shared_state() -> None:
    """Close the global shared state connection."""
    global _shared_state
    if _shared_state:
        await _shared_state.close()
        _shared_state = None


__all__ = [
    "SharedControlPlaneState",
    "AgentState",
    "TaskState",
    "set_shared_state",
    "get_shared_state",
    "get_shared_state_sync",
    "close_shared_state",
]
